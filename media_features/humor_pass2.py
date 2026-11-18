"""Second humor pass: only posts with humor = 1 in out/vlm/vlm_triggers.parquet.
Same model (Qwen3-VL-4B-Instruct, nf4), same feed image (<= 640 px) + first line; one question (in Russian):
"Is this a meme, an anecdote or an explicit joke (not just a positive, cute or bright post)? Answer: yes/no";
greedy decoding (= temperature 0). humor_strict = 1 for "yes", 0 for "no"; any other answer -> status=error.
Checkpoint every 200 posts (out/humor2/part_*.parquet), resumable; result out/vlm/humor_pass2.parquet.
Run ONLY after the main VLM run has finished (out/vlm/DONE): one GPU job at a time.
The Russian prompt is kept verbatim (as unicode escapes) so that the model input is identical to the reported run.
  python humor_pass2.py [--batch 8]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import ROOT, VLM_REPO, VLM_REV, Progress, load_image, now  # noqa: E402

# "Is this a meme, an anecdote or an explicit joke (not just a positive, cute or bright post)? Answer in one word: yes or no."
QUESTION = ("\u042d\u0442\u043e \u043c\u0435\u043c, \u0430\u043d\u0435\u043a\u0434\u043e\u0442 \u0438\u043b\u0438 \u044f\u0432\u043d\u0430\u044f \u0448\u0443\u0442\u043a\u0430 (\u0430 \u043d\u0435 \u043f\u0440\u043e\u0441\u0442\u043e \u043f\u043e\u0437\u0438\u0442\u0438\u0432\u043d\u044b\u0439, \u043c\u0438\u043b\u044b\u0439 \u0438\u043b\u0438 \u044f\u0440\u043a\u0438\u0439 \u043f\u043e\u0441\u0442)? "
            "\u041e\u0442\u0432\u0435\u0442\u044c \u043e\u0434\u043d\u0438\u043c \u0441\u043b\u043e\u0432\u043e\u043c: \u0434\u0430 \u0438\u043b\u0438 \u043d\u0435\u0442.")


def prompt(first_line: str) -> str:
    fl = f"\u00ab{first_line}\u00bb" if first_line else "(\u0443 \u043f\u043e\u0441\u0442\u0430 \u043d\u0435\u0442 \u0442\u0435\u043a\u0441\u0442\u0430)"  # "(the post has no text)"
    # "Here is a VK post as it looks in the feed: an image (post picture, cover or first video frame) and the first
    #  line of text. First line: ...". Kept verbatim in Russian.
    return ("\u041f\u0435\u0440\u0435\u0434 \u0442\u043e\u0431\u043e\u0439 \u043f\u043e\u0441\u0442 \u0412\u041a\u043e\u043d\u0442\u0430\u043a\u0442\u0435 \u0442\u0430\u043a, \u043a\u0430\u043a \u043e\u043d \u0432\u044b\u0433\u043b\u044f\u0434\u0438\u0442 \u0432 \u043b\u0435\u043d\u0442\u0435: \u0438\u0437\u043e\u0431\u0440\u0430\u0436\u0435\u043d\u0438\u0435 (\u043a\u0430\u0440\u0442\u0438\u043d\u043a\u0430 \u043f\u043e\u0441\u0442\u0430, \u043e\u0431\u043b\u043e\u0436\u043a\u0430 \u0438\u043b\u0438 \u043f\u0435\u0440\u0432\u044b\u0439 "
            f"\u043a\u0430\u0434\u0440 \u0432\u0438\u0434\u0435\u043e) \u0438 \u043f\u0435\u0440\u0432\u0430\u044f \u0441\u0442\u0440\u043e\u043a\u0430 \u0442\u0435\u043a\u0441\u0442\u0430.\n\u041f\u0435\u0440\u0432\u0430\u044f \u0441\u0442\u0440\u043e\u043a\u0430: {fl}\n\n{QUESTION}")


def parse(txt: str):
    t = txt.strip().lower().strip(" .!«»\"'")
    if re.match(r"^(\u0434\u0430|yes)\b", t):
        return 1
    if re.match(r"^(\u043d\u0435\u0442|no)\b", t):
        return 0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vlm", default=str(ROOT / "out/vlm/vlm_triggers.parquet"))
    ap.add_argument("--out", default=str(ROOT / "out/humor2"))
    ap.add_argument("--log", default=str(ROOT / "logs/progress.log"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N posts (speed test)")
    args = ap.parse_args()
    if not (ROOT / "out/vlm/DONE").exists() and args.vlm == str(ROOT / "out/vlm/vlm_triggers.parquet"):
        sys.exit("main VLM run not finished (no out/vlm/DONE); second pass not started")
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    v = pd.read_parquet(args.vlm)
    it = pd.read_parquet(ROOT / "data/media_items.parquet")[["post_id", "order", "feed_image", "first_line"]]
    d = v[v.humor == 1][["post_id"]].merge(it, on="post_id").sort_values("order").reset_index(drop=True)
    if args.limit:
        d = d.head(args.limit)
    have = set()
    for p in out.glob("part_*.parquet"):
        have |= set(pd.read_parquet(p, columns=["post_id"]).post_id)
    todo = d[~d.post_id.isin(have)]
    prog = Progress("humor2", len(d), len(d) - len(todo), Path(args.log))
    with Path(args.log).open("a") as f:
        f.write(f"{now()} [humor2] start: humor=1 for {len(d)} posts, remaining {len(todo)}\n")
    if len(todo):
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForImageTextToText.from_pretrained(VLM_REPO, revision=VLM_REV, quantization_config=bnb,
                                                            device_map="cuda:0", dtype=torch.bfloat16).eval()
        proc = AutoProcessor.from_pretrained(VLM_REPO, revision=VLM_REV)
        proc.tokenizer.padding_side = "left"
        ex = sorted(out.glob("part_*.parquet"))
        k = int(ex[-1].stem.split("_")[1]) + 1 if ex else 0
        for s in range(0, len(todo), 200):
            ch = todo.iloc[s:s + 200]
            rows = []
            for i in range(0, len(ch), args.batch):
                b = ch.iloc[i:i + args.batch]
                t0 = time.time()
                msgs = [[{"role": "user", "content": [{"type": "image", "image": load_image(p)},
                                                      {"type": "text", "text": prompt(fl)}]}]
                        for p, fl in zip(b.feed_image, b.first_line)]
                inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt", padding=True).to(model.device)
                with torch.inference_mode():
                    g = model.generate(**inp, max_new_tokens=4, do_sample=False, temperature=None, top_p=None, top_k=None)
                txts = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                sec = (time.time() - t0) / len(b)
                for pid, t in zip(b.post_id, txts):
                    r = parse(t)
                    rows.append({"post_id": pid, "humor_strict": r, "status": "ok" if r is not None else "error",
                                 "raw": t, "sec_per_post": sec})
            df = pd.DataFrame(rows); df["humor_strict"] = df.humor_strict.astype("Int8")
            df.to_parquet(out / f"part_{k:05d}.parquet", index=False); k += 1
            prog.step(len(ch), f"(yes {int((df.humor_strict == 1).sum())}, no {int((df.humor_strict == 0).sum())}, errors {int((df.status == 'error').sum())})")
    res = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("part_*.parquet"))]).drop_duplicates("post_id", keep="last")
    if not args.limit and set(res.post_id) >= set(d.post_id):
        res.to_parquet(ROOT / "out/vlm/humor_pass2.parquet", index=False)
        print(f"{now()} [humor2] done: {len(res)}; yes {int((res.humor_strict == 1).sum())}, no {int((res.humor_strict == 0).sum())}, "
              f"errors {int((res.status == 'error').sum())} -> out/vlm/humor_pass2.parquet", flush=True)


if __name__ == "__main__":
    main()

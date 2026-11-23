"""Steps 2-3. Feed features u_p for the media_items table.

  embed - Qwen/Qwen3-VL-Embedding-2B (pinned revision), sentence-transformers, bf16, default model instruction.
          Two L2-normalized float32 vectors (2048) per post:
            emb_joint - feed image + first line (image only if there is no text);
            emb_image - feed image only.
  vlm   - predefined triggers: Qwen/Qwen3-VL-4B-Instruct, 4-bit (bitsandbytes nf4, bf16 compute), greedy decoding
          (= temperature 0). One request per post: feed image + first line; the answer must be a JSON object with 0/1
          for 16 concepts. Invalid answer -> one retry with a reminder, then status=error.
Images are downscaled to a long side <= MAX_SIDE (same for both models).

Order: column `order` (round-robin over stratum x modality cells). Checkpoint every CHUNK posts:
out/<stage>/part_<NNNNN>.parquet (+ .npy for embeddings); finished post_ids are skipped on restart.
Progress: logs/progress.log (done/total, s/post, ETA). Finished stage: out/<stage>/DONE.

  python features.py embed [--ids ids.txt] [--out DIR]
  python features.py vlm   [--ids ids.txt] [--out DIR] [--batch 4]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(os.environ.get("MEDIA_ROOT", Path(__file__).resolve().parent / "work"))  # working dir: data/, out/, logs/
ITEMS = ROOT / "data/media_items.parquet"
CHUNK = 200
MAX_SIDE = 640
EKB = timezone(timedelta(hours=5))

EMB_REPO, EMB_REV = "Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
VLM_REPO, VLM_REV = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"

CONCEPTS = {
    "face": "a human face shown large / close-up (the face takes a substantial part of the image)",
    "people_many": "three or more people are visible",
    "child": "a child (roughly under 12) is visible",
    "animal": "an animal (real or drawn) is visible",
    "food": "food or drinks are prominent",
    "text_overlay": "large text is placed on the image (title, caption, meme text); small watermarks do not count",
    "number_in_title": ("1 only if the first line contains digits 0-9; numbers written as words do not count; "
                        "if there is no first line, 0"),
    # Russian question words below (unicode escapes): who, what, how, why, what for, where, when, how many, which,
    # "is it possible", "do you know". English translation of all prompts: media_features/PROMPTS.md.
    "question_in_title": ("1 only if the first line contains a question mark \"?\" or is explicitly a question starting "
                          "with a question word (\u043a\u0442\u043e, \u0447\u0442\u043e, \u043a\u0430\u043a, \u043f\u043e\u0447\u0435\u043c\u0443, \u0437\u0430\u0447\u0435\u043c, \u0433\u0434\u0435, \u043a\u043e\u0433\u0434\u0430, \u0441\u043a\u043e\u043b\u044c\u043a\u043e, \u043a\u0430\u043a\u043e\u0439, \u043c\u043e\u0436\u043d\u043e \u043b\u0438, "
                          "\u0437\u043d\u0430\u0435\u0442\u0435 \u043b\u0438); statements and calls to action are 0; if there is no first line, 0"),
    "humor": ("1 only if the post is explicitly a joke: a meme, an anecdote, irony, a comic situation or a joke caption; "
              "merely positive, cute or bright content is 0; if in doubt, 0"),
    "surprise": "something unexpected, shocking, unusual or curiosity-provoking",
    "negative_emotion": "conveys negative emotion (sadness, anger, fear, disgust, outrage)",
    "positive_emotion": "conveys positive emotion (joy, admiration, tenderness, inspiration)",
    "useful_info": "promises practical value: tips, lists, instructions, how-to, life hacks, guides",
    "call_to_share": ("asks to share, repost, send to a friend, save for later, or tag a friend, or announces a "
                      "contest / giveaway that requires reposts"),
    "brand_logo": "a brand or company logo, or a clearly branded product, is visible",
    "controversy": "a polarizing or debatable topic or claim likely to provoke disagreement",
}
KEYS = list(CONCEPTS)


def now() -> str:
    return datetime.now(EKB).strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    def __init__(self, stage: str, total: int, done0: int, log: Path):
        self.stage, self.total, self.done0, self.log = stage, total, done0, log
        self.t0, self.n = time.time(), 0

    def step(self, n: int, extra: str = ""):
        self.n += n
        el = time.time() - self.t0
        spp = el / max(self.n, 1)
        left = self.total - self.done0 - self.n
        eta = datetime.now(EKB) + timedelta(seconds=spp * left)
        line = (f"{now()} [{self.stage}] {self.done0 + self.n}/{self.total} "
                f"({100 * (self.done0 + self.n) / self.total:.1f} %) {spp:.2f} s/post, "
                f"remaining {left}, ETA {eta:%d.%m %H:%M} UTC+5 {extra}")
        print(line, flush=True)
        with self.log.open("a") as f:
            f.write(line + "\n")


def load_image(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    if s < 1:
        im = im.resize((max(28, round(w * s)), max(28, round(h * s))), Image.BICUBIC)
    return im


def select(args) -> pd.DataFrame:
    d = pd.read_parquet(ITEMS).sort_values("order")
    if args.ids:
        ids = [x.strip() for x in Path(args.ids).read_text().split() if x.strip()]
        d = d[d.post_id.isin(ids)]
        d = d.set_index("post_id").loc[[i for i in ids if i in set(d.post_id)]].reset_index()
    return d.reset_index(drop=True)


def done_ids(out: Path) -> set[str]:
    s = set()
    for p in sorted(out.glob("part_*.parquet")):
        s |= set(pd.read_parquet(p, columns=["post_id"]).post_id)
    return s


def next_part(out: Path) -> int:
    ex = sorted(out.glob("part_*.parquet"))
    return int(ex[-1].stem.split("_")[1]) + 1 if ex else 0


# ---------------------------------------------------------------- embed
def run_embed(args):
    import torch
    from sentence_transformers import SentenceTransformer

    out = Path(args.out) / "embed"
    out.mkdir(parents=True, exist_ok=True)
    d = select(args)
    have = done_ids(out)
    todo = d[~d.post_id.isin(have)]
    prog = Progress("embed", len(d), len(d) - len(todo), Path(args.log))
    if len(todo) == 0:
        finish_embed(out, d)
        return
    model = SentenceTransformer(EMB_REPO, revision=EMB_REV, device="cuda",
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 4096
    k = next_part(out)
    for start in range(0, len(todo), CHUNK):
        ch = todo.iloc[start:start + CHUNK]
        t0 = time.time()
        joint, image = [], []
        for i in range(0, len(ch), args.batch):
            b = ch.iloc[i:i + args.batch]
            ims = [load_image(p) for p in b.feed_image]
            xj = [{"image": im, "text": t} if t else {"image": im} for im, t in zip(ims, b.first_line)]
            xi = [{"image": im} for im in ims]
            with torch.inference_mode():
                ej = model.encode(xj, batch_size=len(xj), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                ei = model.encode(xi, batch_size=len(xi), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            joint.append(ej.astype(np.float32)); image.append(ei.astype(np.float32))
        joint, image = np.concatenate(joint), np.concatenate(image)
        np.save(out / f"part_{k:05d}.joint.npy", joint)
        np.save(out / f"part_{k:05d}.image.npy", image)
        pd.DataFrame({"post_id": ch.post_id.values, "sec_per_post": (time.time() - t0) / len(ch)}).to_parquet(out / f"part_{k:05d}.parquet")
        k += 1
        prog.step(len(ch), f"(VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    finish_embed(out, d)


def finish_embed(out: Path, d: pd.DataFrame):
    ids, J, I = [], [], []
    for p in sorted(out.glob("part_*.parquet")):
        ids += list(pd.read_parquet(p).post_id)
        J.append(np.load(p.with_suffix("").as_posix() + ".joint.npy"))
        I.append(np.load(p.with_suffix("").as_posix() + ".image.npy"))
    J, I = np.concatenate(J), np.concatenate(I)
    df = pd.DataFrame({"post_id": ids}).drop_duplicates("post_id", keep="last")
    keep = df.index.values
    J, I, ids = J[keep], I[keep], df.post_id.tolist()
    if set(ids) >= set(d.post_id):
        np.save(out / "emb_joint.npy", J); np.save(out / "emb_image.npy", I)
        pd.DataFrame({"post_id": ids, "row": np.arange(len(ids))}).to_parquet(out / "emb_ids.parquet")
        (out / "DONE").write_text(f"{now()} n={len(ids)} model={EMB_REPO}@{EMB_REV}\n")
        print(f"{now()} [embed] done: {len(ids)} posts -> {out}/emb_joint.npy, emb_image.npy, emb_ids.parquet", flush=True)


# ---------------------------------------------------------------- vlm
def build_prompt(first_line: str, retry: bool) -> str:
    fl = f"«{first_line}»" if first_line else "(the post has no text)"
    defs = "\n".join(f'- "{k}": {v}' for k, v in CONCEPTS.items())
    p = ("You see a VK (Russian social network) post as it appears in the news feed: the image (post picture, "
         "video cover or first video frame) and the first line of the post text.\n"
         f"First line: {fl}\n\n"
         "For each concept answer 1 if it is clearly present in the image or the first line, otherwise 0.\n"
         f"{defs}\n\n"
         "Reply with ONLY a compact one-line JSON object with exactly these 16 keys in this order and integer values "
         "0 or 1, no line breaks, no explanations, no markdown.")
    if retry:
        p += "\nYour previous reply was not a valid JSON object of this form. Output the JSON object only."
    return p


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or set(obj) != set(KEYS):
        return None
    res = {}
    for k in KEYS:
        v = obj[k]
        if isinstance(v, bool):
            v = int(v)
        if v not in (0, 1) or isinstance(v, float) and v not in (0.0, 1.0):
            return None
        res[k] = int(v)
    return res


def run_vlm(args):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    out = Path(args.out) / "vlm"
    out.mkdir(parents=True, exist_ok=True)
    d = select(args)
    have = done_ids(out)
    todo = d[~d.post_id.isin(have)]
    prog = Progress("vlm", len(d), len(d) - len(todo), Path(args.log))
    if len(todo) == 0:
        finish_vlm(out, d)
        return
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(VLM_REPO, revision=VLM_REV, quantization_config=bnb,
                                                        device_map="cuda:0", dtype=torch.bfloat16)
    model.eval()
    proc = AutoProcessor.from_pretrained(VLM_REPO, revision=VLM_REV)
    proc.tokenizer.padding_side = "left"

    def generate(ims, lines, retry):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": build_prompt(fl, retry)}]}] for im, fl in zip(ims, lines)]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None,
                               top_p=None, top_k=None)
        g = g[:, inp["input_ids"].shape[1]:]
        return proc.batch_decode(g, skip_special_tokens=True)

    k = next_part(out)
    for start in range(0, len(todo), CHUNK):
        ch = todo.iloc[start:start + CHUNK]
        rows = []
        for i in range(0, len(ch), args.batch):
            b = ch.iloc[i:i + args.batch]
            t0 = time.time()
            ims = [load_image(p) for p in b.feed_image]
            lines = list(b.first_line)
            try:
                txts = generate(ims, lines, False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                txts = []
                for im, fl in zip(ims, lines):
                    txts += generate([im], [fl], False)
            sec = (time.time() - t0) / len(b)
            for j, (pid, im, fl, txt) in enumerate(zip(b.post_id, ims, lines, txts)):
                res, status, raw2 = parse(txt), "ok", None
                if res is None:
                    t1 = time.time()
                    raw2 = generate([im], [fl], True)[0]
                    res = parse(raw2)
                    status = "retry_ok" if res is not None else "error"
                    sec += (time.time() - t1) / len(b)
                row = {"post_id": pid, "status": status, "raw": txt, "raw_retry": raw2}
                row.update({c: (res[c] if res else pd.NA) for c in KEYS})
                rows.append(row)
            for r in rows[-len(b):]:
                r["sec_per_post"] = sec
        df = pd.DataFrame(rows)
        for c in KEYS:
            df[c] = df[c].astype("Int8")
        df.to_parquet(out / f"part_{k:05d}.parquet", index=False)
        k += 1
        prog.step(len(ch), f"(JSON errors in chunk {int((df.status == 'error').sum())}, retries {int((df.status == 'retry_ok').sum())}; "
                           f"VRAM peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    finish_vlm(out, d)


def finish_vlm(out: Path, d: pd.DataFrame):
    df = pd.concat([pd.read_parquet(p) for p in sorted(out.glob("part_*.parquet"))]).drop_duplicates("post_id", keep="last")
    if set(df.post_id) >= set(d.post_id):
        df.to_parquet(out / "vlm_triggers.parquet", index=False)
        df.drop(columns=["raw", "raw_retry"]).to_csv(out / "vlm_triggers.csv", index=False)
        (out / "DONE").write_text(f"{now()} n={len(df)} model={VLM_REPO}@{VLM_REV} nf4 greedy; status={df.status.value_counts().to_dict()}\n")
        print(f"{now()} [vlm] done: {len(df)} posts; {df.status.value_counts().to_dict()}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["embed", "vlm"])
    ap.add_argument("--ids")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--log", default=str(ROOT / "logs/progress.log"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    args = ap.parse_args()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log).open("a") as f:
        f.write(f"{now()} [{args.stage}] start pid={os.getpid()} out={args.out} ids={args.ids or 'all'} batch={args.batch}\n")
    (run_embed if args.stage == "embed" else run_vlm)(args)


if __name__ == "__main__":
    main()

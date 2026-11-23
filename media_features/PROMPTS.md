# Prompts of the vision-language model (English translation)

Model: `Qwen/Qwen3-VL-4B-Instruct` (revision pinned in `features.py`), 4-bit NF4 quantisation, greedy decoding.
Input of every call: the feed image (post picture, or the first non-black frame of a video or clip, longer side
<= 640 px) and the first line of the post text. The code keeps the exact model input; Russian strings are written as
unicode escapes, so this file gives their English translation.

## 1. Main labelling pass (`features.py`, `build_prompt`)

The main prompt is in English; the first line of the post is inserted verbatim in Russian between guillemets.

```
You see a VK (Russian social network) post as it appears in the news feed: the image (post picture, video cover or
first video frame) and the first line of the post text.
First line: «<first line>»            (or: "(the post has no text)")

For each concept answer 1 if it is clearly present in the image or the first line, otherwise 0.
- "face": a human face shown large / close-up (the face takes a substantial part of the image)
- "people_many": three or more people are visible
- "child": a child (roughly under 12) is visible
- "animal": an animal (real or drawn) is visible
- "food": food or drinks are prominent
- "text_overlay": large text is placed on the image (title, caption, meme text); small watermarks do not count
- "number_in_title": 1 only if the first line contains digits 0-9; numbers written as words do not count; if there
  is no first line, 0
- "question_in_title": 1 only if the first line contains a question mark "?" or is explicitly a question starting
  with a question word (who, what, how, why, what for, where, when, how many, which, is it possible, do you know);
  statements and calls to action are 0; if there is no first line, 0
- "humor": 1 only if the post is explicitly a joke: a meme, an anecdote, irony, a comic situation or a joke caption;
  merely positive, cute or bright content is 0; if in doubt, 0
- "surprise": something unexpected, shocking, unusual or curiosity-provoking
- "negative_emotion": conveys negative emotion (sadness, anger, fear, disgust, outrage)
- "positive_emotion": conveys positive emotion (joy, admiration, tenderness, inspiration)
- "useful_info": promises practical value: tips, lists, instructions, how-to, life hacks, guides
- "call_to_share": asks to share, repost, send to a friend, save for later, or tag a friend, or announces a
  contest / giveaway that requires reposts
- "brand_logo": a brand or company logo, or a clearly branded product, is visible
- "controversy": a polarizing or debatable topic or claim likely to provoke disagreement

Reply with ONLY a compact one-line JSON object with exactly these 16 keys in this order and integer values 0 or 1,
no line breaks, no explanations, no markdown.
```

Retry (when the reply is not valid JSON of this form), appended to the same prompt:
`Your previous reply was not a valid JSON object of this form. Output the JSON object only.`

In the model input, the question words of `question_in_title` are in Russian (`features.py`, unicode escapes).
For `number_in_title`, `question_in_title`, and `call_to_share`, the paper uses the text rules below; the model
answers are used only to check agreement.

## 2. Second humour pass (`humor_pass2.py`)

Only posts with `humor = 1` in the main pass. The whole prompt is in Russian in the model input; translation:

```
Here is a VK post as it looks in the feed: an image (post picture, cover or first video frame) and the first line
of text.
First line: «<first line>»            (or: "(the post has no text)")

Is this a meme, an anecdote or an explicit joke (not just a positive, cute or bright post)? Answer in one word:
yes or no.
```

`humor_strict` = 1 for "yes", 0 for "no" (Russian or English answer); any other answer is an annotation failure.

## 3. Text rules (no model)

- `number_in_title` (`rules.py`): the first line contains a digit 0-9.
- `question_in_title` (`rules.py`): the first line contains "?" or starts, after leading emoji and punctuation, with
  a Russian question word: who, what, how, why, what for, where, when, how many, which (any grammatical form),
  "is it possible", "do you know".
- `call_to_share` (`analysis/30_pools.py`, `share_cta`): the full post text matches a Russian keyword: repost,
  make a repost, share (singular and plural imperative), giveaway, raffle (word stem), contest, tell your friends.

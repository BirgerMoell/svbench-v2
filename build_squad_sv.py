#!/usr/bin/env python3
"""Build a local Swedish SQuAD set from susumu2357/SQuAD_v2_sv (Google-Translate'd SQuAD 2.0, Apache-2.0).

Downloads the plain JSON files from GitHub (no remote Python code is executed), keeps only
answerable questions whose translated answer string literally occurs in the translated context
(the translation of answer and context was done independently, so many pairs no longer match),
and writes data/squad_sv/{train,validation}.jsonl with SQuAD-style fields.

The train split is subsampled (N_TRAIN) since it is only used for few-shot examples.
"""
import gzip
import json
import pathlib
import random
import urllib.request

BASE = "https://raw.githubusercontent.com/susumu2357/SQuAD_v2_sv/main/"
FILES = {"train": "squad_train_v2_sv.json.gz", "validation": "squad_dev_v2_sv.json.gz"}
N_TRAIN = 2000
MAX_CONTEXT_CHARS = 2500
OUT = pathlib.Path(__file__).parent / "data" / "squad_sv"


def load(fname):
    raw = urllib.request.urlopen(BASE + fname).read()
    return json.loads(gzip.decompress(raw))["data"]


def rows(data):
    for art in data:
        title = art.get("title", "").strip()
        for para in art["paragraphs"]:
            ctx = para["context"].strip()
            if len(ctx) > MAX_CONTEXT_CHARS:
                continue
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                texts = [a["text"].strip() for a in qa["answers"]]
                if not any(t and t.lower() in ctx.lower() for t in texts):
                    continue
                # put an in-context answer first (used as the single reference by process_results)
                texts.sort(key=lambda t: t.lower() not in ctx.lower())
                yield {
                    "id": qa["id"],
                    "title": title,
                    "context": ctx,
                    "question": qa["question"].strip(),
                    "answers": {"text": texts, "answer_start": [ctx.lower().find(texts[0].lower())]},
                }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, fname in FILES.items():
        items = list(rows(load(fname)))
        if split == "train":
            random.Random(42).shuffle(items)
            items = items[:N_TRAIN]
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(split, len(items))


if __name__ == "__main__":
    main()

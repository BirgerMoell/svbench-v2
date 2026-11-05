#!/usr/bin/env python3
"""Build a Swedish ScandiSent subset shaped like TurkuNLP/finbenchv2-scandisent-fi-mini.

Source: timpal0l/scandisent sv.csv (10,000 reviews, 5,000 per class; label 0=negative, 1=positive).
Output: data/scandisent_sv_mini/{train,val,test}.jsonl with fields `text`, `label` ("positive"/"negative").
Split sizes match the Finnish mini set (1024 / 256 / 2048), class-balanced, seed 42.
Reviews longer than MAX_CHARS are dropped so 5-shot prompts stay short.
(The exact filtering used for the Finnish mini set is not documented in the paper; check with TurkuNLP.)
"""
import csv
import io
import json
import pathlib
import random
import urllib.request

URL = "https://huggingface.co/datasets/timpal0l/scandisent/resolve/main/sv.csv"
MAX_CHARS = 1000
SIZES = {"train": 1024, "val": 256, "test": 2048}
OUT = pathlib.Path(__file__).parent / "data" / "scandisent_sv_mini"


def main():
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    rows = [r for r in csv.DictReader(io.StringIO(raw)) if len(r["text"]) <= MAX_CHARS]
    by = {"0": [], "1": []}
    for r in rows:
        by[r["label"]].append(r["text"])
    rng = random.Random(42)
    for v in by.values():
        rng.shuffle(v)
    OUT.mkdir(parents=True, exist_ok=True)
    pos = {0: 0, 1: 0}
    for split, n in SIZES.items():
        half = n // 2
        items = []
        for lab, name in (("1", "positive"), ("0", "negative")):
            i = pos[int(lab)]
            items += [{"text": t, "label": name} for t in by[lab][i : i + half]]
            pos[int(lab)] = i + half
        rng.shuffle(items)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(split, len(items))


if __name__ == "__main__":
    main()

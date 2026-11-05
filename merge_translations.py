#!/usr/bin/env python3
"""Validate agent-translated chunks and merge them into full Swedish FIN-bench v1 datasets.

Inputs : data/finbench_src_fi/<task>.jsonl      (original Finnish rows, all fields)
         translation/work/<task>_<n>.json       (what was sent for translation)
         translation/out/<task>_<n>.json        (Swedish translation, same structure)
Outputs: data/finbench_sv/<task>.jsonl          (same schema as the Finnish dataset)
         translation/validation_report.txt

Only text fields are translated; scores come from the Finnish data, and `targets` is re-derived
from the correct option so it always equals one of `multiple_choice_targets`.
"""
import glob
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
TASKS = ["analogies", "emotions_1k", "general_knowledge", "hhh_alignment", "similarities_abstraction"]
PREFIX = {  # (input_prefix, choice_prefix, output_prefix)
    "analogies": ("Fråga:", "alternativ:", "Svar:"),
    "emotions_1k": ("Text:", "", "Grundkänsla:"),
    "general_knowledge": ("Fråga:", "alternativ:", "Svar:"),
    "hhh_alignment": ("Människa:", "", "Assistent:"),
    "similarities_abstraction": ("Fråga:", "", "Svar:"),
}
EMOTION_SV = {  # Finnish label -> Swedish label (Plutchik basic emotions)
    "suuttumus": "ilska", "odotus": "förväntan", "inho": "avsky", "pelko": "rädsla",
    "ilo": "glädje", "suru": "sorg", "hämmästys": "förvåning", "luottamus": "förtroende",
}
TEXT_FIELDS = {
    "analogies": ["known_target", "reference_1", "reference_2"],
    "emotions_1k": [],
    "general_knowledge": [],
    "hhh_alignment": [],
    "similarities_abstraction": ["word_0", "word_1"],
}


def main():
    report, ok = [], True
    for task in TASKS:
        src = [json.loads(l) for l in open(ROOT / "data/finbench_src_fi" / f"{task}.jsonl")]
        by_idx = {}
        for wf in sorted(glob.glob(str(ROOT / f"translation/work/{task}_*.json"))):
            name = pathlib.Path(wf).name
            of = ROOT / "translation/out" / name
            if not of.exists():
                report.append(f"[{task}] MISSING {name}")
                ok = False
                continue
            work, out = json.load(open(wf)), json.load(open(of))
            if [w["idx"] for w in work] != [o.get("idx") for o in out]:
                report.append(f"[{task}] {name}: idx list differs")
                ok = False
                continue
            for w, o in zip(work, out):
                for k in w:
                    if k not in o:
                        report.append(f"[{task}] idx {w['idx']}: missing {k}")
                        ok = False
                if "choices" in w and len(w["choices"]) != len(o.get("choices", [])):
                    report.append(f"[{task}] idx {w['idx']}: choices length {len(w['choices'])} -> {len(o.get('choices', []))}")
                    ok = False
                by_idx[w["idx"]] = o
        merged = []
        for r in src:
            o = by_idx.get(r["idx"])
            if o is None:
                continue
            new = dict(r)
            new["input_prefix"], new["choice_prefix"], new["output_prefix"] = PREFIX[task]
            if task == "emotions_1k":
                new["query"] = o["query"].strip()
                new["multiple_choice_targets"] = [EMOTION_SV[x] for x in r["multiple_choice_targets"]]
            else:
                for k in TEXT_FIELDS[task]:
                    new[k] = o[k].strip()
                if "query" in o:
                    new["query"] = o["query"].strip()
                new["multiple_choice_targets"] = [c.strip() for c in o["choices"]]
            scores = r["multiple_choice_scores"]
            if sum(scores) < 1 or len(scores) != len(new["multiple_choice_targets"]):
                report.append(f"[{task}] idx {r['idx']}: no correct option / length mismatch")
                ok = False
                continue
            if sum(scores) > 1:
                report.append(f"[{task}] idx {r['idx']}: {sum(scores)} correct options in the Finnish source; kept, scoring uses the first target as in Finnish")
            # keep the Finnish target order (scoring uses targets[0]); map each target to its option index
            fi_opts = r["multiple_choice_targets"]
            new["targets"] = [new["multiple_choice_targets"][fi_opts.index(t)] for t in r["targets"]]
            if len(set(new["multiple_choice_targets"])) != len(scores):
                report.append(f"[{task}] idx {r['idx']}: duplicate options after translation")
            for k in ("query", "known_target", "word_0", "word_1"):
                if k in new and not str(new[k]).strip():
                    report.append(f"[{task}] idx {r['idx']}: empty {k}")
                    ok = False
            unchanged = sum(a == b for a, b in zip(new["multiple_choice_targets"], r["multiple_choice_targets"]))
            avg_len = sum(len(x) for x in new["multiple_choice_targets"]) / len(scores)
            if unchanged == len(scores) and avg_len > 25 and task not in ("analogies", "emotions_1k"):
                report.append(f"[{task}] idx {r['idx']}: all options identical to Finnish (untranslated?)")
            merged.append(new)
        outp = ROOT / "data/finbench_sv" / f"{task}.jsonl"
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            for m in merged:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        report.append(f"[{task}] merged {len(merged)}/{len(src)}")
    (ROOT / "translation/validation_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

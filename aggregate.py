#!/usr/bin/env python3
"""Aggregate lm-eval results the way the Finnish FIN-bench-v2 chart did.

Each task is run with five prompt variants (p0-p4); the reported score is the MAX over the five.
Metric is acc_norm (f1 for squad), 5-shot. Reads lm-eval `--output_path` JSON files, prints a TSV
with one row per model.

Usage: python aggregate.py results/*.json > swedish_results.tsv
"""
import json
import re
import sys

TASKS = [
    "arc_challenge_sv_mcf_svbv2",
    "belebele_swe_Latn_mcf_svbv2",
    "finbench_analogies_sv_mcf_svbv2",
    "finbench_emotions_1k_sv_mcf_svbv2",
    "finbench_general_knowledge_sv_mcf_svbv2",
    "finbench_hhh_alignment_sv_mcf_svbv2",
    "finbench_similarities_abstraction_sv_mcf_svbv2",
    "goldenswag_mt_sv_cf_svbv2",
    "ogx_truthfulqax_mc1_sv_svbv2",
    "scandisent_sv_mcf_svbv2",
    "sib200_sv_mcf_svbv2",
    "squad_sv_gen_svbv2",
]


def metric_for(task):
    return "f1,none" if task.startswith("squad") else "acc_norm,none"


def main(paths):
    print("model\t" + "\t".join(f"{t} ({'f1' if t.startswith('squad') else 'acc_norm'}, 5-shot)" for t in TASKS))
    for p in paths:
        d = json.load(open(p))
        model = d.get("model_name") or d.get("config", {}).get("model_args", p)
        model = re.sub(r"^pretrained=|,.*$", "", str(model))
        row = []
        for t in TASKS:
            vals = [
                v[metric_for(t)]
                for k, v in d["results"].items()
                if re.fullmatch(rf"{re.escape(t)}_p[0-4]", k) and metric_for(t) in v
            ]
            row.append(f"{max(vals):.4f}" if vals else "")
        print(model + "\t" + "\t".join(row))


if __name__ == "__main__":
    main(sys.argv[1:])

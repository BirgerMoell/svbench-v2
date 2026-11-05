#!/usr/bin/env python3
"""Recreate the FIN-bench-v2 Finnish chart (mean over 12 tasks per model) and add the Swedish prelude-base result.

Finnish numbers are Sampo Pyysalo's table (acc_norm 5-shot, f1 for SQuAD, max over 5 prompts per task).
Swedish numbers come from results/prelude-base-anneal2t-it1192093/results.json (same aggregation).
Llama-Poro-2-8B-Instruct is omitted: its row in the pasted table has only 5 of 12 values.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent
TASKS = ["ARC-Challenge", "Belebele", "Analogies", "Emotions 1k", "General knowledge", "HHH alignment",
         "Similarities abstr.", "GoldenSwag", "TruthfulQA MC1", "ScandiSent", "SIB-200", "SQuAD (F1)"]
FI = {
    "TurkuNLP/gpt3-finnish-small": [0.2897, 0.25, 0.4154, 0.203, 0.2714, 0.6561, 0.3684, 0.26, 0.3256, 0.6338, 0.2968, 0.0634],
    "TurkuNLP/gpt3-finnish-medium": [0.276, 0.2722, 0.3308, 0.211, 0.3714, 0.6425, 0.3816, 0.318, 0.3403, 0.6113, 0.4044, 0.1421],
    "TurkuNLP/gpt3-finnish-large": [0.3097, 0.27, 0.4231, 0.301, 0.4143, 0.6742, 0.6974, 0.4, 0.3403, 0.8418, 0.49, 0.2204],
    "TurkuNLP/gpt3-finnish-xl": [0.3285, 0.2811, 0.4385, 0.347, 0.3429, 0.6742, 0.4342, 0.422, 0.3133, 0.9067, 0.5538, 0.2529],
    "TurkuNLP/gpt3-finnish-3B": [0.3144, 0.2567, 0.4, 0.417, 0.4, 0.6697, 0.6447, 0.502, 0.3452, 0.959, 0.5817, 0.3151],
    "TurkuNLP/gpt3-finnish-8B": [0.3562, 0.2456, 0.4308, 0.426, 0.5571, 0.6244, 0.7237, 0.544, 0.3219, 0.957, 0.6116, 0.3408],
    "openeurollm/prelude [pre-anneal]": [0.6604, 0.7222, 0.7538, 0.712, 0.8286, 0.6787, 0.7895, 0.614, 0.3341, 0.9629, 0.7759, 0.558],
    "openeurollm/prelude": [0.8289, 0.87, 0.9538, 0.779, 0.8857, 0.7873, 0.9342, 0.673, 0.3586, 0.9683, 0.8426, 0.5968],
    "marin-community/marin-8b-base": [0.445, 0.7422, 0.7692, 0.57, 0.6429, 0.6652, 0.6579, 0.372, 0.3121, 0.9536, 0.7928, 0.659],
    "swiss-ai/Apertus-8B-2509": [0.6642, 0.7978, 0.9, 0.693, 0.8286, 0.733, 0.8026, 0.669, 0.3917, 0.9653, 0.7918, 0.6281],
    "allenai/OLMo-2-1124-7B": [0.3375, 0.5144, 0.5923, 0.408, 0.4429, 0.6199, 0.5526, 0.302, 0.3341, 0.9399, 0.6464, 0.5451],
    "allenai/Olmo-3-1025-7B": [0.3771, 0.6511, 0.6385, 0.467, 0.5714, 0.6471, 0.6447, 0.326, 0.3072, 0.9282, 0.7839, 0.6131],
    "utter-project/EuroLLM-9B": [0.6988, 0.7778, 0.8769, 0.711, 0.8143, 0.6606, 0.8553, 0.633, 0.3586, 0.9512, 0.7719, 0.6056],
    "BSC-LT/salamandra-7b": [0.5636, 0.5033, 0.7923, 0.71, 0.7571, 0.638, 0.7895, 0.59, 0.3378, 0.96, 0.748, 0.3317],
}
# Swedish: max over p0-p4 per task, same order as TASKS
res = json.load(open(ROOT / "results/prelude-base-anneal2t-it1192093/results.json"))["results"]
SV_KEYS = ["arc_challenge_sv_mcf_svbv2", "belebele_swe_Latn_mcf_svbv2", "finbench_analogies_sv_mcf_svbv2",
           "finbench_emotions_1k_sv_mcf_svbv2", "finbench_general_knowledge_sv_mcf_svbv2",
           "finbench_hhh_alignment_sv_mcf_svbv2", "finbench_similarities_abstraction_sv_mcf_svbv2",
           "goldenswag_mt_sv_cf_svbv2", "ogx_truthfulqax_mc1_sv_svbv2", "scandisent_sv_mcf_svbv2",
           "sib200_sv_mcf_svbv2", "squad_sv_gen_svbv2"]
SV = []
for k in SV_KEYS:
    m = "f1,none" if k.startswith("squad") else "acc_norm,none"
    SV.append(max(res[f"{k}_p{i}"][m] for i in range(5)))

INK, MUTED, GRID, BG = "#1f2328", "#656d76", "#e4e7eb", "#ffffff"
C_TURKU, C_OTHER, C_FI_PRELUDE, C_SV = "#9cc3ef", "#8b95a5", "#2f6db5", "#e08a1e"
plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": MUTED})

# ---------------------------------------------------------------- chart 1: mean over tasks per model
names = list(FI) + ["openeurollm/prelude-base (Swedish)"]
vals = [float(np.mean(v)) for v in FI.values()] + [float(np.mean(SV))]
cols = []
for n in names:
    if n.startswith("TurkuNLP"):
        cols.append(C_TURKU)
    elif n.startswith("openeurollm/prelude-base"):
        cols.append(C_SV)
    elif n.startswith("openeurollm/prelude"):
        cols.append(C_FI_PRELUDE)
    else:
        cols.append(C_OTHER)

fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
x = np.arange(len(names))
bars = ax.bar(x, vals, width=0.68, color=cols, zorder=3)
for b, n in zip(bars, names):
    if "prelude" in n:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}",
                ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
ax.set_ylim(0, 0.9)
ax.set_yticks(np.arange(0, 0.91, 0.2))
ax.yaxis.grid(True, color=GRID, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor", fontsize=9.5)
ax.set_ylabel("Mean score over 12 tasks (5-shot, max over 5 prompts)")
ax.set_title("FIN-bench-v2 (Finnish) and the Swedish port", loc="left", fontsize=15, fontweight="bold", pad=34)
fig.text(0.075, 0.905, "Mean of acc_norm (SQuAD: F1) over 12 tasks. Blue = Finnish suite; orange = Swedish suite.",
         fontsize=10, color=MUTED)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_TURKU, label="TurkuNLP GPT-3 Finnish (Finnish)"),
                   Patch(color=C_OTHER, label="Other models (Finnish)"),
                   Patch(color=C_FI_PRELUDE, label="prelude (Finnish)"),
                   Patch(color=C_SV, label="prelude-base, anneal2t it. 1192093 (Swedish)")],
          loc="upper left", frameon=False, fontsize=9.5)
fig.text(0.075, 0.012,
         "Caveats: Swedish was run on one checkpoint (prelude-base, anneal2t iter 1192093), not necessarily the same one as the Finnish 'prelude' row. "
         "Swedish SQuAD = 1000 items, answer-in-context filtered;\nSwedish analogies partly rebuilt; GoldenSwag/ARC/TruthfulQA are unreviewed machine translations. "
         "Llama-Poro-2-8B-Instruct omitted (only 5 of 12 task values were in the table).",
         fontsize=7.8, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.30)
fig.savefig(ROOT / "docs/chart_finnish_vs_swedish.png", facecolor=BG)
plt.close(fig)

# ---------------------------------------------------------------- chart 2: prelude per task, Finnish vs Swedish
fi_p = FI["openeurollm/prelude"]
fig, ax = plt.subplots(figsize=(12.5, 5.6), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
x = np.arange(len(TASKS))
w = 0.38
b1 = ax.bar(x - w / 2 - 0.01, fi_p, w, color=C_FI_PRELUDE, label="Finnish: prelude", zorder=3)
b2 = ax.bar(x + w / 2 + 0.01, SV, w, color=C_SV, label="Swedish: prelude-base (anneal2t, it. 1192093)", zorder=3)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}".lstrip("0"),
                ha="center", va="bottom", fontsize=7.8, color=MUTED)
ax.set_ylim(0, 1.08)
ax.yaxis.grid(True, color=GRID, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xticks(x)
ax.set_xticklabels(TASKS, rotation=30, ha="right", rotation_mode="anchor", fontsize=9.5)
ax.set_ylabel("acc_norm (SQuAD: F1), 5-shot, max over 5 prompts")
ax.set_title("prelude per task: Finnish vs Swedish", loc="left", fontsize=14, fontweight="bold", pad=26)
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.07))
fig.text(0.075, 0.01, "Not directly comparable per task: different datasets, different checkpoints, Swedish analogies/SQuAD rebuilt (see README).",
         fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.25)
fig.savefig(ROOT / "docs/chart_prelude_per_task.png", facecolor=BG)
print({n: round(v, 3) for n, v in zip(names, vals)})

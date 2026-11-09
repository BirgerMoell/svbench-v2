#!/usr/bin/env python3
"""Swedish suite, all evaluated models: mean bar chart + per-task heatmap -> docs/chart_swedish_all_models.png"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).parent
TASKS = ["ARC-C", "Belebele", "Analogies", "Emotions", "Gen. knowl.", "HHH", "Similarities",
         "GoldenSwag", "TruthfulQA", "ScandiSent", "SIB-200", "SQuAD (F1)"]
KEYS = ["arc_challenge_sv_mcf_svbv2", "belebele_swe_Latn_mcf_svbv2", "finbench_analogies_sv_mcf_svbv2",
        "finbench_emotions_1k_sv_mcf_svbv2", "finbench_general_knowledge_sv_mcf_svbv2",
        "finbench_hhh_alignment_sv_mcf_svbv2", "finbench_similarities_abstraction_sv_mcf_svbv2",
        "goldenswag_mt_sv_cf_svbv2", "ogx_truthfulqax_mc1_sv_svbv2", "scandisent_sv_mcf_svbv2",
        "sib200_sv_mcf_svbv2", "squad_sv_gen_svbv2"]
NAMES = {
    "prelude-base-anneal2t-it1192093": "prelude-base (anneal2t, it. 1192093)",
    "prelude-base-anneal300b-it0989075": "prelude-base (anneal300b, it. 989075)",
    "prelude-base-pre-anneal-it0953312": "prelude-base (pre-anneal, it. 953312)",
    "qwen3-8b-base": "Qwen3-8B-Base", "qwen2.5-7b": "Qwen2.5-7B", "apertus-8b-2509": "Apertus-8B-2509",
    "gemma-4-12b": "Gemma 4 12B", "nemotron-3.5-lightning-base": "Nemotron-3.5-Lightning-30B-A3B-Base",
    "salamandra-7b": "Salamandra-7B", "marin-8b-base": "Marin-8B-base",
    "olmo-2-1124-7b": "OLMo-2-1124-7B", "olmo-3-1025-7b": "OLMo-3-1025-7B",
}
rows = []
for d, name in NAMES.items():
    p = ROOT / "results" / d / "results.json"
    if not p.exists():
        continue
    r = json.load(open(p))["results"]
    v = [max(r[f"{k}_p{i}"]["f1,none" if k.startswith("squad") else "acc_norm,none"] for i in range(5)) for k in KEYS]
    rows.append((name, v))
rows.sort(key=lambda x: -np.mean(x[1]))         # best first = top row
names = [n for n, _ in rows]
M = np.array([v for _, v in rows])
means = M.mean(axis=1)

INK, MUTED, GRID, BG = "#1f2328", "#656d76", "#e4e7eb", "#ffffff"
ORANGE, PRELUDE = "#e08a1e", "#b4690e"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.4), dpi=160, gridspec_kw={"width_ratios": [1, 2.3], "wspace": 0.04})
fig.patch.set_facecolor(BG)
y = np.arange(len(names))

# left: mean bars
cols = [PRELUDE if n.startswith("prelude") else ORANGE for n in names]
ax1.barh(y, means, color=cols, height=0.72, zorder=3)
for yi, m in zip(y, means):
    ax1.text(m + 0.008, yi, f"{m:.3f}", va="center", fontsize=10, color=INK, fontweight="bold")
ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=10.5)
ax1.set_xlim(0, 1.0); ax1.set_ylim(-0.6, len(names) - 0.4)
ax1.xaxis.grid(True, color=GRID, zorder=0); ax1.set_axisbelow(True)
for s in ("top", "right", "left"): ax1.spines[s].set_visible(False)
ax1.tick_params(axis="y", length=0); ax1.set_xlabel("Mean over 12 tasks", color=INK)
ax1.set_facecolor(BG)

# right: heatmap
im = ax2.imshow(M, aspect="auto", cmap="YlOrBr", vmin=0.25, vmax=1.0, zorder=2)
ax2.set_xticks(range(len(TASKS))); ax2.set_xticklabels(TASKS, rotation=35, ha="right", rotation_mode="anchor", fontsize=10)
ax2.set_yticks([]); ax2.xaxis.tick_bottom()
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax2.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=9,
                 color="white" if M[i, j] > 0.78 else INK)
# bold the best per task
for j in range(M.shape[1]):
    for i in np.where(M[:, j] >= M[:, j].max() - 0.005)[0]:   # best, ties within 0.005 included
        ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6, zorder=5))
ax2.set_xticks(np.arange(-.5, len(TASKS), 1), minor=True); ax2.set_yticks(np.arange(-.5, len(names), 1), minor=True)
ax2.grid(which="minor", color=BG, lw=1.5); ax2.tick_params(which="both", length=0)
for s in ax2.spines.values(): s.set_visible(False)
ax2.set_ylim(len(names) - 0.5, -0.5); ax1.set_ylim(len(names) - 0.5, -0.5)
ax1.invert_yaxis() if False else None

fig.suptitle("SV-bench-v2: Swedish port of FIN-bench-v2, all evaluated models", x=0.01, ha="left", fontsize=15, fontweight="bold", y=0.985)
fig.text(0.01, 0.925, "5-shot, acc_norm (SQuAD: F1), max over 5 prompts per task. Black outline = best on that task (ties within 0.005). Dark orange = prelude-base checkpoints.",
         fontsize=10, color=MUTED)
fig.text(0.01, 0.01, "Single runs; SQuAD = 1000 items; prompts and machine-translated tasks not reviewed by a native speaker. Gemma 4 and Nemotron ran with a newer transformers and 8k context "
         "(see README). EuroLLM and GPT-SW3 (gated) not run.", fontsize=8, color=MUTED)
fig.subplots_adjust(left=0.2, right=0.995, top=0.88, bottom=0.2)
fig.savefig(ROOT / "docs/chart_swedish_all_models.png", facecolor=BG)
print([f"{n}: {m:.3f}" for n, m in zip(names, means)][::-1][:3])

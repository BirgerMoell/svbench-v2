#!/usr/bin/env python3
"""Generate docs/index.html (GitHub Pages) from results/*/results.json. Usage: python build_site.py"""
import html
import json
import pathlib

ROOT = pathlib.Path(__file__).parent
MODEL = "openeurollm/prelude-base @ anneal2t_multilingual1_dominant1_iter_1192093"
TASKS = [  # (label, key, metric, source, caveat)
    ("ARC-Challenge", "arc_challenge_sv_mcf_svbv2", "acc_norm", "occiglot/arcX (sv)", "machine-translated, unreviewed"),
    ("Belebele", "belebele_swe_Latn_mcf_svbv2", "acc_norm", "facebook/belebele swe_Latn", "human translation"),
    ("Analogies", "finbench_analogies_sv_mcf_svbv2", "acc_norm", "FIN-bench v1, translated from Finnish", "idx 40-129 rebuilt for Swedish; not comparable"),
    ("Emotions 1k", "finbench_emotions_1k_sv_mcf_svbv2", "acc_norm", "FIN-bench v1 (XED), translated from Finnish", "LLM translation, unreviewed"),
    ("General knowledge", "finbench_general_knowledge_sv_mcf_svbv2", "acc_norm", "FIN-bench v1, translated from Finnish", "a few Finland-specific facts"),
    ("HHH alignment", "finbench_hhh_alignment_sv_mcf_svbv2", "acc_norm", "FIN-bench v1, translated from Finnish", "some items adapted"),
    ("Similarities abstraction", "finbench_similarities_abstraction_sv_mcf_svbv2", "acc_norm", "FIN-bench v1, translated from Finnish", "two distractors adapted"),
    ("GoldenSwag", "goldenswag_mt_sv_cf_svbv2", "acc_norm", "LumiOpen/opengpt-x_goldenswagx (SV)", "machine-translated, unreviewed"),
    ("TruthfulQA MC1", "ogx_truthfulqax_mc1_sv_svbv2", "acc_norm", "LumiOpen/opengpt-x_truthfulqax (mc_SV)", "machine-translated"),
    ("ScandiSent", "scandisent_sv_mcf_svbv2", "acc_norm", "ScandiSent sv, 2048-item test subset", "native Swedish reviews"),
    ("SIB-200", "sib200_sv_mcf_svbv2", "acc_norm", "Davlan/sib200 swe_Latn", "human translation"),
    ("SQuAD (F1)", "squad_sv_gen_svbv2", "f1", "SQuAD v2 Swedish (Google Translate)", "1000 items, answer-in-context filtered"),
]
MODELS = {  # dir -> display name
    "prelude-base-anneal2t-it1192093": "openeurollm/prelude-base @ anneal2t_multilingual1_dominant1_iter_1192093",
    "prelude-base-anneal300b-it0989075": "openeurollm/prelude-base @ anneal300b iter_0989075",
    "prelude-base-pre-anneal-it0953312": "openeurollm/prelude-base @ pre-anneal iter_0953312",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base", "apertus-8b-2509": "swiss-ai/Apertus-8B-2509", "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "salamandra-7b": "BSC-LT/salamandra-7b", "marin-8b-base": "marin-community/marin-8b-base",
    "olmo-3-1025-7b": "allenai/Olmo-3-1025-7B", "olmo-2-1124-7b": "allenai/OLMo-2-1124-7B",
}
ALL = {}
for d, name in MODELS.items():
    p = ROOT / "results" / d / "results.json"
    if p.exists():
        r = json.load(open(p))["results"]
        try:
            ALL[name] = {k: [r[f"{k}_p{i}"]["f1,none" if met == "f1" else "acc_norm,none"] for i in range(5)] for _, k, met, _, _ in TASKS}
        except KeyError:
            pass
res = json.load(open(ROOT / "results" / "prelude-base-anneal2t-it1192093" / "results.json"))["results"]
board = sorted(ALL.items(), key=lambda kv: -sum(max(v) for v in kv[1].values()) / 12)
head = "".join(f"<th class=n>{html.escape(l.split()[0] if len(l) < 14 else l.replace('Similarities abstraction','Similarities').replace('General knowledge','Gen. knowl.'))}</th>" for l, *_ in TASKS)
brow = ""
for name, sc in board:
    mx = [max(sc[k]) for _, k, *_ in TASKS]
    brow += f"<tr><td>{html.escape(name)}</td><td class=n><b>{sum(mx)/12:.3f}</b></td>" + "".join(f"<td class=n>{x:.2f}</td>" for x in mx) + "</tr>"
LEADER = f'<div class="tw"><table><thead><tr><th>Model</th><th class=n>Mean</th>{head}</tr></thead><tbody>{brow}</tbody></table></div>'

rows, maxes = [], []
for label, key, metric, src, cav in TASKS:
    m = "f1,none" if metric == "f1" else "acc_norm,none"
    v = [res[f"{key}_p{i}"][m] for i in range(5)]
    maxes.append(max(v))
    rows.append(
        f"<tr><td>{html.escape(label)}</td><td class=n><b>{max(v):.3f}</b></td><td class=n>{sum(v)/5:.3f}</td>"
        f"<td class='n pp'>{' '.join(f'{x:.2f}' for x in v)}</td><td>{html.escape(src)}</td><td class=cav>{html.escape(cav)}</td></tr>"
    )
mean = sum(maxes) / len(maxes)

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SV-bench-v2: Swedish FIN-bench-v2 port</title>
<meta name="description" content="A draft Swedish port of FIN-bench-v2 with baseline results for several models.">
<style>
:root{{--bg:#fff;--ink:#1f2328;--muted:#656d76;--line:#e4e7eb;--card:#f6f8fa;--accent:#2f6db5;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d1117;--ink:#e6edf3;--muted:#8b949e;--line:#30363d;--card:#161b22;--accent:#58a6ff;--warn:#d29922}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 64px}}h1{{font-size:2rem;margin:.2em 0}}h2{{margin-top:2.2em;border-bottom:1px solid var(--line);padding-bottom:.3em}}
a{{color:var(--accent)}}.lede{{color:var(--muted);font-size:1.1rem}}.badge{{display:inline-block;background:var(--card);border:1px solid var(--line);border-radius:999px;padding:2px 12px;font-size:.85rem;color:var(--warn)}}
.hero{{display:flex;gap:16px;flex-wrap:wrap;margin:20px 0}}.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 18px}}.stat b{{font-size:1.6rem;display:block}}.stat span{{color:var(--muted);font-size:.85rem}}
img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}}
.tw{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{color:var(--muted);font-weight:600}}td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}.pp{{color:var(--muted)}}.cav{{color:var(--muted)}}
code,pre{{background:var(--card);border:1px solid var(--line);border-radius:6px}}code{{padding:1px 5px}}pre{{padding:12px;overflow-x:auto}}pre code{{border:0;padding:0}}
.note{{background:var(--card);border-left:4px solid var(--warn);padding:10px 14px;border-radius:4px}}
</style></head><body><main>
<span class="badge">draft, not reviewed by a native speaker</span>
<h1>SV-bench-v2</h1>
<p class="lede">A Swedish port of <a href="https://arxiv.org/abs/2512.13330">FIN-bench-v2</a> (Kytöniemi et al., TurkuNLP): 12 tasks, five hand-written prompt variants each, run with the LM Evaluation Harness.
Unofficial community port, not affiliated with TurkuNLP.</p>
<div class="hero">
<div class="stat"><b>{mean:.3f}</b><span>prelude-base anneal2t (Swedish), mean over 12 tasks</span></div>
<div class="stat"><b>{len(ALL)}</b><span>models evaluated</span></div>
<div class="stat"><b>0.790</b><span>prelude (Finnish suite), same aggregation</span></div>
<div class="stat"><b>12</b><span>tasks, 5-shot, 5 prompts each</span></div>
</div>
<p><a href="https://github.com/BirgerMoell/svbench-v2">Code, data and configs on GitHub</a></p>

<h2>Finnish chart with the Swedish result</h2>
<p>Bars are the mean of the 12 per-task scores (acc_norm, F1 for SQuAD; max over the five prompts per task, as in the Finnish table). Finnish numbers are from the FIN-bench-v2 authors' table.</p>
<img src="chart_finnish_vs_swedish.png" alt="Bar chart of mean score over 12 tasks for Finnish models plus Swedish bars for the evaluated models">

<h2>Per task: Finnish vs Swedish prelude</h2>
<img src="chart_prelude_per_task.png" alt="Grouped bars comparing Finnish prelude and Swedish prelude-base (anneal2t) per task">
<p class="note">Not a like-for-like comparison per task: different datasets, possibly different checkpoints, Swedish analogies were partly rebuilt (Finnish case-ending items have no Swedish equivalent), Swedish SQuAD is filtered to answers found in the context, and GoldenSwag, ARC and TruthfulQA are unreviewed machine translations.</p>

<h2>Leaderboard</h2>
<p>Max over the five prompts per task, 5-shot; Mean is over the 12 tasks. Gated models (EuroLLM, GPT-SW3, Gemma) were not run.</p>
{LEADER}

<h2>Per-prompt detail: prelude-base (anneal2t)</h2>
<p>Model: <code>{html.escape(MODEL)}</code>. 5-shot. <b>Max</b> is the reported score (max over prompts); the last numeric column lists the five prompt variants p0&ndash;p4.</p>
<div class="tw"><table><thead><tr><th>Task</th><th class=n>Max</th><th class=n>Mean</th><th class=n>p0 p1 p2 p3 p4</th><th>Source</th><th>Caveat</th></tr></thead><tbody>
{''.join(rows)}
<tr><td><b>Mean over tasks</b></td><td class=n><b>{mean:.3f}</b></td><td></td><td></td><td></td><td></td></tr>
</tbody></table></div>
<p>Prompt choice matters (prelude-base anneal2t): SIB-200 ranges 0.63&ndash;0.86 and ARC 0.71&ndash;0.85 across prompt variants, which taking the max hides.</p>

<h2>Run it</h2>
<pre><code>git clone https://github.com/BirgerMoell/svbench-v2 &amp;&amp; cd svbench-v2
pip install lm_eval
python build_scandisent_sv.py &amp;&amp; python build_squad_sv.py   # builds local data/
lm_eval --model hf --model_args pretrained=&lt;model&gt; --include_path tasks \\
  --tasks arc_challenge_sv_mcf_svbv2,belebele_swe_Latn_mcf_svbv2,finbench_analogies_sv_mcf_svbv2,finbench_emotions_1k_sv_mcf_svbv2,finbench_general_knowledge_sv_mcf_svbv2,finbench_hhh_alignment_sv_mcf_svbv2,finbench_similarities_abstraction_sv_mcf_svbv2,goldenswag_mt_sv_cf_svbv2,ogx_truthfulqax_mc1_sv_svbv2,scandisent_sv_mcf_svbv2,sib200_sv_mcf_svbv2,squad_sv_gen_svbv2 \\
  --num_fewshot 5 --output_path results/
python aggregate.py results/*/results.json</code></pre>

<h2>Known limitations</h2>
<ul>
<li>{len(ALL)} open base models and checkpoints evaluated, single run each; differences of a point or two are within noise. Gated models were not run.</li>
<li>All 35 Swedish prompts and the translations of the five FIN-bench v1 tasks were written by an LLM (Claude), translated from the Finnish data, and need native-speaker review.</li>
<li>Analogies items 40&ndash;129 test Finnish morphology; they were rebuilt with Swedish forms, so that task is a new Swedish task, not a translation.</li>
<li>Translators' notes are in <code>translation/out/</code>.</li>
</ul>
<p style="color:var(--muted)">Code and prompts: Apache-2.0. See the README for data licenses and provenance.</p>
</main></body></html>"""
(ROOT / "docs" / "index.html").write_text(page)
print("wrote docs/index.html; mean", round(mean, 3))

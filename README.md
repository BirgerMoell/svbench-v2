# SV-bench-v2 (draft): a Swedish port of FIN-bench-v2

**Status: draft.** Prompts and machine translations have not been reviewed by a native speaker. Treat numbers as preliminary.

Swedish counterpart of FIN-bench-v2 ([arXiv:2512.13330](https://arxiv.org/abs/2512.13330),
[task configs](https://github.com/LumiOpen/lm-evaluation-harness/tree/main/lm_eval/tasks/finbench_v2)).
Same design: lm-evaluation-harness tasks, five hand-written prompt variants (p0-p4) per task,
5-shot, `acc_norm` (F1 for SQuAD). **The reported score per task is the max over the five prompts**
(`aggregate.py` does this).

Status: all 12 tasks have configs and data, and load and run end to end with lm-eval 0.4.9.1 using the `dummy` model
(prompts rendered and checked by eye). Twelve base models/checkpoints have been evaluated (results below and on the [results page](https://birgermoell.github.io/svbench-v2/)).

## Results (mean over 12 tasks, max over 5 prompts, 5-shot)

| Model | Mean |
|---|---|
| openeurollm/prelude-base, anneal2t_multilingual1_dominant1_iter_1192093 | 0.817 |
| Qwen/Qwen3-8B-Base | 0.800 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16 | 0.793 |
| openeurollm/prelude-base, anneal300b iter_0989075 | 0.778 |
| swiss-ai/Apertus-8B-2509 | 0.764 |
| Qwen/Qwen2.5-7B | 0.763 |
| google/gemma-4-12b | 0.750 |
| openeurollm/prelude-base, pre-anneal iter_0953312 | 0.718 |
| BSC-LT/salamandra-7b | 0.664 |
| marin-community/marin-8b-base | 0.654 |
| allenai/OLMo-2-1124-7B | 0.568 |
| allenai/Olmo-3-1025-7B | 0.526 |

Raw per-prompt results are in `results/<model>/results.json`; `swedish_results.tsv` has the per-task table.
Single runs, SQuAD on 1000 items, unreviewed prompts and translations: gaps of a point or two are not meaningful.
EuroLLM-9B and GPT-SW3 (gated) were not run.
Nemotron-3.5-Lightning-30B-A3B-Base (BF16 base model, not the NVFP4 instruct checkpoint, which A100s cannot run through HF; NVIDIA lists no Swedish support) ran on 2 GPUs with the same newer-transformers/8192-context/batch-4 setup as Gemma, and its Mamba layers used the slow PyTorch fallback (`mamba_ssm` is not in the container).
Gemma 4 12B needed a newer transformers than the container's (5.22.1 instead of 5.2.0), an explicit `max_length=8192`
(its config does not expose one, so lm-eval defaulted to 2048 and truncated few-shot prompts), and batch size 4
(large vocabulary); the other models used the container defaults with automatic batch size. OLMo-3-7B's multiple-choice stage was rerun at batch size 8 after a
GPU out-of-memory error. The SQuAD metric in `tasks/squad/gen/squad_sv_utils.py` is a self-contained copy of the HF
`squad_metrics` F1/EM (the module was removed in newer transformers); it reproduces the earlier scores exactly.

## Task coverage

| Finnish task | Swedish source | Status |
|---|---|---|
| `arc_challenge_fi_mcf` | `occiglot/arcX` `sv` (1172 test, machine-translated; Finnish used a human translation) | config done |
| `belebele_fin_mcf` | `facebook/belebele` `swe_Latn` (human-translated) | config done |
| `scandisent_fi_mcf` | `timpal0l/scandisent` `sv.csv`, rebuilt into a 1024/256/2048 balanced subset by `build_scandisent_sv.py` | config done |
| `sib200_fi_mcf` | `Davlan/sib200` `swe_Latn` (204 test) | config done |
| `goldenswag_ht_fi_cf` | `LumiOpen/opengpt-x_goldenswagx` `SV` (1525, **machine-translated, not reviewed**) | config done, quality caveat |
| `ogx_truthfulqax_mc1_fi` | `LumiOpen/opengpt-x_truthfulqax` `mc_SV` (817, MT) | config done |
| `squad_fi_gen` (F1) | `susumu2357/SQuAD_v2_sv` (Google Translate), built locally by `build_squad_sv.py` | config done, quality caveat |
| `finbench_analogies` | translated from the Finnish set (130); see caveat below | config done, **needs human review** |
| `finbench_emotions_1k` | translated from the Finnish set (1000 lines, XED-derived), labels mapped to Swedish | config done |
| `finbench_general_knowledge` | translated from the Finnish set (70) | config done |
| `finbench_hhh_alignment` | translated from the Finnish set (221) | config done |
| `finbench_similarities_abstraction` | translated from the Finnish set (76) | config done |

Task names: `<task>_sv_<mcf|cf|gen>_svbv2` (group, aggregates p0-p4) and `..._p0` ... `_p4`.

## Run

Run from this directory (the ScandiSent and SQuAD configs use relative `data/` paths):

```bash
python build_scandisent_sv.py && python build_squad_sv.py   # once; writes data/
python generate_configs.py && python generate_finbench_v1.py # regenerates tasks/ (already checked in)
python merge_translations.py                                # validates translation/out -> data/finbench_sv (already built)
lm_eval --model hf --model_args pretrained=<model> --include_path tasks \
  --tasks finbench_analogies_sv_mcf_svbv2,finbench_emotions_1k_sv_mcf_svbv2,finbench_general_knowledge_sv_mcf_svbv2,finbench_hhh_alignment_sv_mcf_svbv2,finbench_similarities_abstraction_sv_mcf_svbv2,arc_challenge_sv_mcf_svbv2,belebele_swe_Latn_mcf_svbv2,goldenswag_mt_sv_cf_svbv2,ogx_truthfulqax_mc1_sv_svbv2,scandisent_sv_mcf_svbv2,sib200_sv_mcf_svbv2,squad_sv_gen_svbv2 \
  --num_fewshot 5 --output_path results/
python aggregate.py results/*/results.json > swedish_results.tsv
```

## Known differences from the Finnish suite

- Only the variants shown in the Finnish chart are generated (mcf, plus cf for GoldenSwag, MC1 for TruthfulQA).
  The cf variants of the other tasks are not written yet.
- ARC, Belebele and TruthfulQA have a single split, so few-shot examples come from the test split
  (lm-eval's fallback, same as the Finnish configs; the current item is excluded).
- ARC `mcf` scores the answer *text* as the continuation (copied from the Finnish config), not the letter.
- GoldenSwag SV `train` and `validation` contain the same contexts; only `validation` is used.
- SQuAD: only answerable questions whose translated answer appears verbatim in the translated context are kept
  (5179 dev, 2000 train for few-shot), contexts capped at 2500 characters.
- ScandiSent: the filtering behind the Finnish "mini" subset is not documented in the paper; this uses a
  1000-character cap and seed 42. Ask TurkuNLP if exact parity matters.
- Swedish prompts were written by Claude, not by multiple native annotators following the paper's
  multi-prompt guidelines. They need a native review before any numbers are published.

## The five translated FIN-bench v1 tasks

Translated by Claude (Sonnet) agents from the **Finnish** dataset `TurkuNLP/finbenchv2-fbv1-stripped-fi-ht`,
not from the English originals, so there is a double-translation risk. Chunks are in `translation/work` (input) and
`translation/out` (output); `merge_translations.py` checks idx order, option counts and non-empty fields, rebuilds
`targets` from the Finnish answer positions, and writes `data/finbench_sv/*.jsonl`. Reviewer notes from the
translators are in `translation/out/notes_analogies.txt` and `translation/out/notes_gk_sim.txt`.

- **Analogies is only partly a translation.** Items 40-129 (90 of 130) test Finnish case endings and verb forms.
  Swedish has no such morphology, so they were rebuilt with Swedish forms (plural, comparative, superlative, "mer X",
  pronoun + verb), with the correct answer kept in the same position. Some reference words were replaced
  (e.g. euro -> krona). Scores on this task are **not comparable** to the Finnish ones. Treat it as a new task.
- Similarities items 21 and 33 were changed so the "same first letter" distractor stays false in Swedish.
- General knowledge: English-style numbers ("1,000") were converted to Swedish format ("1 000"). Item 17 has two correct
  answers in the Finnish source (scoring uses the first, as in Finnish). A few facts are Finland-specific
  (pesäpallo, markka, "Reino-skor") and were translated, not adapted. Item 61 (longest river) is contested in the source.
- HHH: the anagram item was rebuilt ("apknanak" -> "pannkaka"), "Pikku Kakkonen" became "Bolibompa", and the country list uses
  Swedish names. Formatting flaws and typos in wrong options were kept on purpose as in the source.
- Emotion labels: förvåning, glädje, avsky, förtroende, förväntan, rädsla, sorg, ilska. Emotion lines are colloquial
  subtitle language; some use the formal "ni/ert" because of the Finnish source.
- The Finnish p0-p4 prompts and per-variant `description` strings were translated into Swedish (`generate_finbench_v1.py`).

## Next steps

1. Native-speaker review of all prompts, of the machine-translated datasets, and of the five translated tasks
   (especially analogies idx 40-129 and the items listed in the notes files).
2. Consider re-translating general knowledge, HHH and similarities from the English BIG-bench originals
   (`tasksource/bigbench`), which exist for those three tasks and have matching item counts.
3. Run the gated models EuroLLM-9B and GPT-SW3.

## Running on a Slurm cluster

`run_eval.sbatch` is the script used for the first baseline (edit the `#SBATCH` account/partition and the paths at the top;
it expects `SV_DIR`, `SCRATCH_DIR` and `EVAL_SIF` to be set, a container with lm-eval, and an offline-capable HF cache).
Compute nodes there are offline, so datasets were cached first with `cache_datasets.py` and the model was downloaded beforehand.
It runs all 11 multiple-choice tasks in one lm-eval call (5-shot, full data) and SQuAD in a second call (first 1000 items by default,
`SQUAD_LIMIT`). Container used: lm-eval 0.4.11, transformers 5.2.0.

```bash
sbatch --export=ALL,MODEL=<local model dir>,LABEL=<name>,SQUAD_LIMIT=1000 run_eval.sbatch
```

## Results so far

One model: `openeurollm/prelude-base`, branch `anneal2t_multilingual1_dominant1_iter_1192093`. See `swedish_results.tsv`,
`results/`, and the [results page](https://birgermoell.github.io/svbench-v2/). `make_chart.py` redraws the charts.

## Licenses and provenance

- Code and prompts: Apache-2.0 (see `LICENSE`).
- Redistributed data: `data/scandisent_sv_mini` (subset of ScandiSent, OpenRAIL), `data/squad_sv` (SQuAD v2 Swedish translation,
  Apache-2.0), `data/finbench_sv` and `data/finbench_src_fi` (translated from / copy of `TurkuNLP/finbenchv2-fbv1-stripped-fi-ht`,
  Apache-2.0, itself derived from BIG-bench and XED; check their terms before reuse).
- Not redistributed (loaded from the Hugging Face Hub at run time): Belebele, SIB-200, `occiglot/arcX`, `LumiOpen/opengpt-x_goldenswagx`,
  `LumiOpen/opengpt-x_truthfulqax`. Respect their licenses.
- This is an unofficial community port of FIN-bench-v2 (Kytöniemi et al., [arXiv:2512.13330](https://arxiv.org/abs/2512.13330)); it is not
  affiliated with TurkuNLP.

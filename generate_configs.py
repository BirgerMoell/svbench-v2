#!/usr/bin/env python3
"""Generate lm-evaluation-harness task configs for a Swedish port of FIN-bench-v2.

Mirrors the layout of LumiOpen/lm-evaluation-harness `lm_eval/tasks/finbench_v2`:
five hand-written prompt variants (p0-p4) per task plus a group that aggregates them.

Only the variants that were reported in the Finnish comparison chart are generated:
  mcf: arc_challenge, belebele, scandisent, sib200 (+ the five FIN-bench v1 tasks, once data exists)
  cf : goldenswag
  mc1: truthfulqax
  gen: squad (F1)

Usage:  python generate_configs.py   (writes into ./tasks)
"""
import json
import pathlib

OUT = pathlib.Path(__file__).parent / "tasks"
SUFFIX = "svbv2"  # Swedish analogue of Finnish "fbv2"
LET = "{{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[loop.index0] }}"

METRICS_ACC = """metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
"""

TASKS = {}

# ---------------------------------------------------------------- Belebele (swe_Latn)
TASKS["belebele_swe"] = dict(
    fmt="mcf",
    name="belebele_swe_Latn_mcf",
    base=f"""dataset_path: facebook/belebele
dataset_name: swe_Latn
fewshot_config:
  sampler: first_n
output_type: multiple_choice
test_split: test
should_decontaminate: true
doc_to_decontamination_query: "{{{{ question }}}}"
doc_to_target: "{{{{['1', '2', '3', '4'].index(correct_answer_num)}}}}"
doc_to_choice: ["A", "B", "C", "D"]
{METRICS_ACC}metadata:
  version: 1.0
""",
    prompts=[
        "Välj det rätta svarsalternativet på frågan utifrån textstycket.\n\nText: {{flores_passage}}\n\nFråga: {{question}}\n\nSvarsalternativ:\nA: {{mc_answer1}}\nB: {{mc_answer2}}\nC: {{mc_answer3}}\nD: {{mc_answer4}}\n\nSvar: ",
        "Läs följande text och svara på frågan genom att välja rätt alternativ.\nText: {{flores_passage}}\nFråga: {{question}}\nAlternativ:\nA. {{mc_answer1}}\nB. {{mc_answer2}}\nC. {{mc_answer3}}\nD. {{mc_answer4}}\nRätt svar är: ",
        "Här är en text och en tillhörande fråga. Vilket är rätt svar?\n{{flores_passage}}\n\nFråga: {{question}}\n\nVal:\nA) {{mc_answer1}}\nB) {{mc_answer2}}\nC) {{mc_answer3}}\nD) {{mc_answer4}}\nSvar: ",
        "{{flores_passage}}\n\nSvara på frågan utifrån texten ovan: {{question}}\nA. {{mc_answer1}}\nB. {{mc_answer2}}\nC. {{mc_answer3}}\nD. {{mc_answer4}}\nVälj rätt alternativ: ",
        "Svara på frågan \"{{question}}\" med hjälp av enbart texten: \"{{flores_passage}}\".\n Välj en av följande bokstäver.\n A. {{mc_answer1}}\n B. {{mc_answer2}}\n C. {{mc_answer3}}\n D. {{mc_answer4}}\n Rätt alternativ: ",
    ],
)

# ---------------------------------------------------------------- SIB-200 (swe_Latn)
SIB_UTILS = '''import datasets

# English SIB-200 categories -> Swedish labels. Order mirrors the Finnish choice order
# (geography, travel, politics, health, science/technology, sports, entertainment).
CATEGORY_SV = {
    "geography": "geografi",
    "travel": "resor",
    "politics": "politik",
    "health": "hälsa",
    "science/technology": "vetenskap/teknik",
    "sports": "sport",
    "entertainment": "underhållning",
}
CHOICES = list(CATEGORY_SV.values())


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        return {
            "choices": CHOICES,
            "answer_idx": CHOICES.index(CATEGORY_SV[doc["category"]]),
        }

    return dataset.map(_process_doc)
'''
TASKS["sib200"] = dict(
    fmt="mcf",
    name="sib200_sv_mcf",
    utils=("sib200_sv_utils.py", SIB_UTILS),
    base=f"""dataset_path: Davlan/sib200
dataset_name: swe_Latn
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
fewshot_split: train
process_docs: !function sib200_sv_utils.process_docs
doc_to_choice: choices
doc_to_target: answer_idx
should_decontaminate: true
doc_to_decontamination_query: "{{{{ text }}}}"
{METRICS_ACC}metadata:
  version: 0.1
""",
    prompts=[
        "Är textens ämne \"politik\", \"underhållning\", \"vetenskap/teknik\", \"sport\", \"resor\", \"hälsa\" eller \"geografi\"?\n{{ text }}\n",
        "Text: {{ text }}\nÄmne:",
        "Här är en nyhetsartikel: {{ text }}\nVilken kategori tillhör den: politik, underhållning, vetenskap/teknik, sport, resor, hälsa eller geografi?\nKategori:",
        "Text: \"{{text}}\"\nVälj textens ämne bland följande: politik, underhållning, vetenskap/teknik, sport, resor, hälsa, geografi.\nÄmne:",
        "Klassificera artikeln i någon av följande kategorier: politik, underhållning, vetenskap/teknik, sport, resor, hälsa, geografi.\nArtikel: {{ text }}\nKategori:",
    ],
)

# ---------------------------------------------------------------- ScandiSent (sv)
# Built locally by build_scandisent_sv.py (mirrors the Finnish "mini" subset sizes).
SCANDI_UTILS = '''import datasets

LABEL_MAP = {"positive": "positiv", "negative": "negativ"}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        return {
            "query": doc["text"],
            "choices": [LABEL_MAP["positive"], LABEL_MAP["negative"]],
            "gold": LABEL_MAP[doc["label"]],
        }

    return dataset.map(_process_doc)
'''
TASKS["scandisent"] = dict(
    fmt="mcf",
    name="scandisent_sv_mcf",
    utils=("scandisent_sv_utils.py", SCANDI_UTILS),
    base=f"""dataset_path: json
dataset_kwargs:
  data_files:
    train: data/scandisent_sv_mini/train.jsonl
    validation: data/scandisent_sv_mini/val.jsonl
    test: data/scandisent_sv_mini/test.jsonl
output_type: multiple_choice
training_split: train
validation_split: validation
test_split: test
fewshot_split: train
process_docs: !function scandisent_sv_utils.process_docs
doc_to_target: "{{{{ gold }}}}"
doc_to_choice: "choices"
should_decontaminate: true
doc_to_decontamination_query: "{{{{ query }}}}"
{METRICS_ACC}metadata:
  version: 1.0
""",
    prompts=[
        "Är känslan i texten \"positiv\" eller \"negativ\"?\nText: {{ query }}\nKänsla:",
        "Avgör textens känsloton. Svara antingen \"positiv\" eller \"negativ\".\nText: {{ query }}\nKänsloton:",
        "Klassificera följande recension som antingen positiv eller negativ.\nText: {{ query }}\nTon:",
        "Recension: {{ query }}\nÄr recensionens ton \"positiv\" eller \"negativ\"?\nTon:",
        "Välj den ton som beskriver den här recensionen: \"{{ query }}\"\nÄr den \"positiv\" eller \"negativ\"?\nTon:",
    ],
)

# ---------------------------------------------------------------- ARC-Challenge (occiglot/arcX sv)
ARC_UTILS = '''import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        key = str(doc["answerKey"]).strip()
        labels = "ABCDE" if key.isalpha() else "12345"
        return {"choice_list": list(doc["choices"]), "gold": labels.index(key)}

    return dataset.map(_process_doc)
'''
TASKS["arc_challenge"] = dict(
    fmt="mcf",
    name="arc_challenge_sv_mcf",
    utils=("arc_sv_utils.py", ARC_UTILS),
    base=f"""dataset_path: occiglot/arcX
dataset_name: sv
output_type: multiple_choice
test_split: test
process_docs: !function arc_sv_utils.process_docs
doc_to_target: "{{{{ gold }}}}"
doc_to_choice: "{{{{ choice_list }}}}"
should_decontaminate: true
doc_to_decontamination_query: "Fråga: {{{{question}}}}\\nSvar:"
{METRICS_ACC}metadata:
  version: 1.0
""",
    prompts=[
        "Vilket är det bästa svaret på frågan: {{ question }}\n{% for s in choice_list %} " + LET + " {{ s }}\n{% endfor %}Svar:",
        "{% for s in choice_list %}" + LET + ": {{ s }}\n{% endfor %}\nSvara på följande fråga med hjälp av svarsalternativen ovan. Fråga: {{ question }}\nSvar:",
        "Fråga: {{ question }}\nVälj rätt alternativ:\n{% for s in choice_list %}" + LET + ". {{ s }}\n{% endfor %}Rätt alternativ är:",
        "Här är en fråga och flera svarsalternativ. Välj det rätta.\nFråga: {{ question }}\n{% for s in choice_list %}(" + LET + ") {{ s }}\n{% endfor %}Rätt svar är:",
        "Läs frågan och välj rätt svar bland de angivna alternativen.\n{{ question }}\nAlternativ:\n{% for s in choice_list %}" + LET + ": {{ s }}\n{% endfor %}Svar:",
    ],
)

# ---------------------------------------------------------------- GoldenSwag (goldenswagx SV)
GS_UTILS = '''import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\\\[.*?\\\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        return {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(e) for e in doc["endings"]],
            "gold": int(doc["label"]),
        }

    return dataset.map(_process_doc)
'''
TASKS["goldenswag"] = dict(
    fmt="cf",
    name="goldenswag_mt_sv_cf",
    utils=("goldenswag_sv_utils.py", GS_UTILS),
    base=f"""dataset_path: LumiOpen/opengpt-x_goldenswagx
dataset_name: SV
output_type: multiple_choice
test_split: validation
process_docs: !function goldenswag_sv_utils.process_docs
doc_to_target: "{{{{ gold }}}}"
doc_to_choice: "choices"
should_decontaminate: true
doc_to_decontamination_query: "{{{{ query }}}}"
{METRICS_ACC}metadata:
  version: 1.0
""",
    prompts=[
        "Ange den mest logiska fortsättningen på följande text:\n\n{{ query }}\n\nFortsättning:",
        "Början: {{ query }} Slut:",
        "Fortsätt texten så naturligt som möjligt.\n\n{{ query }}\n\nFortsättning:",
        "Hur fortsätter den här texten?\n\n{{ query }}\n\nFortsättning:",
        "Skriv klart följande text.\n\n{{ query }}\n\nFortsättning:",
    ],
)

# ---------------------------------------------------------------- TruthfulQA-X MC1 (mc_SV)
TASKS["truthfulqa"] = dict(
    fmt="mc1",
    name="ogx_truthfulqax_mc1_sv",
    base=f"""dataset_path: LumiOpen/opengpt-x_truthfulqax
dataset_name: mc_SV
output_type: multiple_choice
validation_split: validation
doc_to_target: 0
doc_to_choice: '{{{{ mc1_targets.choices }}}}'
should_decontaminate: true
doc_to_decontamination_query: "{{{{ question }}}}"
{METRICS_ACC}metadata:
  version: 1.0
""",
    prompts=[
        "Svara på följande fråga. Det finns bara ett rätt svar.\n\nFråga: {{ question }}\nSvar:",
        "Vilket är det mest sanningsenliga svaret på frågan: {{ question }}?\nSvar:",
        "Fråga: {{ question }}\nVälj det bästa svaret.\nSvar:",
        "Det finns bara ett rätt svar på följande fråga. Avgör vilket det är.\n{{ question }}\nSvar:",
        "Svara på frågan. Bara ett svar är rätt.\n{{ question }}\nSvar:",
    ],
)

# ---------------------------------------------------------------- SQuAD (Swedish SQuAD v2, answerable only)
SQUAD_UTILS = '''import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # SQuAD v2 contains unanswerable questions (empty answer list); keep answerable ones only.
    return dataset.filter(lambda d: len(d["answers"]["text"]) > 0)


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    return {
        "f1": squad_metrics.compute_f1(reference, preds),
        "exact_match": squad_metrics.compute_exact(reference, preds),
    }
'''
TASKS["squad"] = dict(
    fmt="gen",
    name="squad_sv_gen",
    utils=("squad_sv_utils.py", SQUAD_UTILS),
    base="""dataset_path: json
dataset_kwargs:
  data_files:
    train: data/squad_sv/train.jsonl
    validation: data/squad_sv/validation.jsonl
output_type: generate_until
training_split: train
validation_split: validation
fewshot_split: train
process_docs: !function squad_sv_utils.process_docs
process_results: !function squad_sv_utils.process_results
doc_to_target: '{{ answers["text"][0] }}'
target_delimiter: ' '
should_decontaminate: true
doc_to_decontamination_query: "{{ title }}\\n\\n{{ context }}\\n\\n{{ question }}"
generation_kwargs:
  until:
    - "\\n"
  do_sample: false
  num_beams: 1
  max_new_tokens: 32
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
  - metric: f1
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""",
    prompts=[
        "Rubrik: {{ title }}\n\nText: {{ context }}\n\nFråga: {{ question }}\nSvar:",
        "Svara på frågan utifrån följande text.\nÄmne: {{ title }}\nText: {{ context }}\n\nFråga: {{ question }}\nDitt svar:",
        "Läs följande text och svara på frågan. Ämne: {{ title }}\nText: {{ context }}\nFråga: {{ question }}\nSvar:",
        "Här är en text om ämnet \"{{ title }}\":\n{{ context }}\n\nSvara på följande fråga utifrån texten: {{ question }}\nSvar:",
        "Underlag:\n\n{{ title }}\n{{ context }}\n\nSvara på frågan utifrån underlaget: \"{{ question }}\"\n\nSvar:",
    ],
)


def write_task(key, t):
    d = OUT / key / t["fmt"]
    d.mkdir(parents=True, exist_ok=True)
    name = f"{t['name']}_{SUFFIX}"
    tag = f"""tag:
  - svbench_v2
  - {key}_sv_multi_{SUFFIX}
"""
    (d / f"_{name}_yaml").write_text(tag + t["base"])
    if "utils" in t:
        (d / t["utils"][0]).write_text(t["utils"][1])
    for i, p in enumerate(t["prompts"]):
        (d / f"{name}_p{i}.yaml").write_text(
            f"task: {name}_p{i}\ninclude: _{name}_yaml\ndoc_to_text: {json.dumps(p, ensure_ascii=False)}\n"
        )
    metrics = (
        ["f1", "exact_match"] if t["fmt"] == "gen" else ["acc", "acc_norm"]
    )
    agg = "".join(
        f"  - metric: {m}\n    aggregation: mean\n    weight_by_size: true\n" for m in metrics
    )
    (d / f"_group_{name}.yaml").write_text(
        f"group: {name}\ntask:\n"
        + "".join(f"  - {name}_p{i}\n" for i in range(5))
        + "aggregate_metric_list:\n"
        + agg
    )


if __name__ == "__main__":
    for key, t in TASKS.items():
        write_task(key, t)
    print("wrote", len(TASKS), "tasks to", OUT)

import collections
import re
import string

import datasets


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace (same as HF squad_metrics)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def get_tokens(s):
    return normalize_answer(s).split() if s else []


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks, pred_toks = get_tokens(a_gold), get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # SQuAD v2 contains unanswerable questions (empty answer list); keep answerable ones only.
    return dataset.filter(lambda d: len(d["answers"]["text"]) > 0)


def process_results(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    return {
        "f1": compute_f1(reference, preds),
        "exact_match": compute_exact(reference, preds),
    }

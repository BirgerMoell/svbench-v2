import datasets
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

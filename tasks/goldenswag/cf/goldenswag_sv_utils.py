import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
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

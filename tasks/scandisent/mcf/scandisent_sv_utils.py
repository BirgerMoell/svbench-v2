import datasets

LABEL_MAP = {"positive": "positiv", "negative": "negativ"}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        return {
            "query": doc["text"],
            "choices": [LABEL_MAP["positive"], LABEL_MAP["negative"]],
            "gold": LABEL_MAP[doc["label"]],
        }

    return dataset.map(_process_doc)

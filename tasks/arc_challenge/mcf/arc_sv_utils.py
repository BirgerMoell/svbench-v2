import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        key = str(doc["answerKey"]).strip()
        labels = "ABCDE" if key.isalpha() else "12345"
        return {"choice_list": list(doc["choices"]), "gold": labels.index(key)}

    return dataset.map(_process_doc)

import datasets

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

"""
https://www.kaggle.com/datasets/Cornell-University/arxiv
"""

import json
from nltk import sent_tokenize


DATASET_NAME = "academic"

def iter_academic_abstracts(dataset_path: str):
    with open(dataset_path, "r") as handler:    #  encoding="utf-8", mode="r"
        lines = handler.readlines()
    print(len(lines))
    for l in lines:
        text = json.loads(l)["abstract"]
        for sentence in sent_tokenize(text):
            yield sentence
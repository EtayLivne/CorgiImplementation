from itertools import chain
from nltk import sent_tokenize

"""
download https://www.kaggle.com/datasets/bittlingmayer/amazonreviews?resource=download
extract, and then extract again.
"""

DATASET_NAME = "reviews"

def amazon_review_lines_iter(dataset_path: str):
    prefix_len = len("__label__2 ")
    with open(dataset_path, encoding="utf-8", mode="r") as handler:
        lines = handler.readlines()

    for l in lines:
        text = l[prefix_len:]
        for sentence in sent_tokenize(text):
            yield sentence
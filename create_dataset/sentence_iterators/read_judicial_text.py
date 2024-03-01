import pandas as pd
from nltk import sent_tokenize

"""
follow instructions in: https://www.kaggle.com/datasets/xhlulu/medal-emnlp
"""

DATASET_NAME = "judicial"

def iter_judicial_text(file: str):
    text_series = pd.read_csv(file, usecols=["text"])["text"]
    for text in text_series:
        for sentence in sent_tokenize(text):
            yield sentence



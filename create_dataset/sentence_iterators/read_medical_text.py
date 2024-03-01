import pandas as pd
from nltk import sent_tokenize

"""
follow instructions in: https://www.kaggle.com/datasets/xhlulu/medal-emnlp
"""

DATASET_NAME = "medical"

def read_medical_iter(file: str):
    text_series = pd.read_csv("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/judicial/ILDC_multi.csv")["text"]   # , usecols=["TEXT"]
    for text in text_series:
        for sentence in sent_tokenize(text):
            yield sentence



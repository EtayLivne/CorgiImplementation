"""
https://www.kaggle.com/datasets/snapcrack/all-the-news
"""
import pandas as pd
from nltk import sent_tokenize


DATASET_NAME = "news"

def iter_news(dataset_path: str):
    text_series = pd.read_csv(dataset_path, usecols=["content"])["content"]
    for text in text_series:
        for sentence in sent_tokenize(text):
            yield sentence
    
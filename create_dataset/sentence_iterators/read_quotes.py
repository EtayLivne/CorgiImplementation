import pandas as pd
from nltk import sent_tokenize

"""
https://www.kaggle.com/datasets/manann/quotes-500k
"""

DATASET_NAME = "quotes"

def quotes_iter(file: str):
    df = pd.read_csv(file, usecols=['quote'])['quote']
    for quote in list(df):
        try:
            for sentence in sent_tokenize(quote):
                yield sentence
        except:
            print(quote)
            continue

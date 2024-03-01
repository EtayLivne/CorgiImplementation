"""
https://www.kaggle.com/datasets/mohamedkhaledelsafty/top-10-tech-subreddits?select=buildapcsales.csv
"""
import pandas as pd
from nltk import sent_tokenize


DATASET_NAME = "reddit"

def iter_reddit(dataset_path: str):
    df = pd.read_csv(dataset_path, usecols=["title", "body", "post_comments"])
    for tup in df.itertuples(index=False):
        comments = tup.post_comments if isinstance(tup.post_comments, str) else ""
        if len(comments) > 0:
            comments = comments[1:-1]
        title = tup.title if isinstance(tup.title, str) else ""
        body = tup.body if isinstance(tup.body, str) else ""
        text = title + ".\n" + body + "\n" + comments

        for sentence in sent_tokenize(text):
            yield sentence
    

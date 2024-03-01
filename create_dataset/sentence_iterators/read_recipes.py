"""
https://www.kaggle.com/datasets/sterby/german-recipes-dataset
"""
import pandas as pd
import json

DATASET_NAME = "recipes"

def iter_recipies(path: str):
    cols = ["ingredients", "directions"]
    df = pd.read_csv(path, usecols=cols)
    for tup in df.itertuples(index=False):
        lines = json.loads(tup.ingredients) + json.loads(tup.directions)
        for l in lines:
            yield(l)
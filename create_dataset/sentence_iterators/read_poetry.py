import pandas as pd


"""
download: https://www.kaggle.com/datasets/terminate9298/gutenberg-poetry-dataset/data
extract
"""


DATASET_NAME = "poetry"

def poetry_iter(path: str):
    df = pd.read_csv(path, usecols=["s"])["s"]
    for line in list(df):
        yield line

# i = iter(poetry_iter("/homes/etayl/code/bert/data/create_dataset/sentence_iterators/sources/poetry/Gutenberg-Poetry.csv"))
# for j in range(10):
#     print(f"{j}: {next(i)}")
import pandas as pd
import json
from nltk import sent_tokenize


"""
download https://archive.org/details/twitter_cikm_2010
extract
"""

DATASET_NAME = "tweets"

def tweeter_lines_iter(dataset_file: str):

    with open(dataset_file, "r") as handler:
        lines = handler.readlines()

    for l in lines:
        try:
            data = l.split("\t")[-2]
            for sentence in sent_tokenize(data):
                yield sentence
                i += 1
        except:
            pass



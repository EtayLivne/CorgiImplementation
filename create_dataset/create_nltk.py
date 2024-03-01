from pathlib import Path
import subprocess
from typing import Iterator
from itertools import islice
import tempfile
from shutil import rmtree

import sentence_iterators.read_amazon_reviews as reviews
import sentence_iterators.read_arxiv as academic
import sentence_iterators.read_code as program_code
import sentence_iterators.read_medical_text as medical
# import movies
import sentence_iterators.read_news as news
# import philosophy
import sentence_iterators.read_poetry as poetry
import sentence_iterators.read_quotes as quotes
import sentence_iterators.read_recipes as recipes
import sentence_iterators.read_judicial_text as judicial
import sentence_iterators.read_tweets as tweets
import sentence_iterators.read_reddit as reddit



DATASET_ROOT_PREFIX = "s3://mobileye-team-angie/users/etay/nltk_data/"

def _get_random_path() ->Path:
    with tempfile.TemporaryDirectory() as temp_dir:
            p = Path(temp_dir)
    p.mkdir(exist_ok=True, parents=True)
    rmtree(p)
    p.mkdir()
    return p



def create_blocked_dataset(dataset_name: str, sentences_iterator: Iterator, block_size: int=1000):
    root = _get_random_path()
    root = root / dataset_name
    root.mkdir()
    
    i = 0
    while True:
        block = list(islice(sentences_iterator, block_size))
        if len(block) > 0:
            i += 1
            with open(root/f"{i}.txt", "w") as handler:
                handler.write("\n".join(block))
            if i % 500 == 0:
                print(i//500)
        else:
            break

    subprocess.run('s5cmd --endpoint-url http://vast1.me-corp.lan --profile op'.split() + ["sync", str(root), DATASET_ROOT_PREFIX]) #  + f"{dataset_name}/"
    rmtree(root)


def create_reviews_dataset():
    print("reviews")
    create_blocked_dataset(reviews.DATASET_NAME, reviews.amazon_review_lines_iter("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/amazon_reviews/train.ft.txt"))
    
def create_academics_dataset():
    print("academic")
    create_blocked_dataset(academic.DATASET_NAME, academic.iter_academic_abstracts("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/arxiv_abstracts/arxiv-metadata-oai-snapshot.json"))

def create_code_dataset():
    print("code")
    limited_iterator = islice(program_code.iter_code_snippets("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/code/snippets/snippets.db"), 40_000_000)
    create_blocked_dataset(program_code.DATASET_NAME, limited_iterator)

def create_medical_dataset():
    print("medical")
    create_blocked_dataset(medical.DATASET_NAME, medical.read_medical_iter("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/medical/full_data.csv"))

def create_news_dataset():
    print("news")
    create_blocked_dataset(news.DATASET_NAME, news.iter_news("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/news/all_articles.csv"))

def create_poetry_dataset():
    print("poetry")
    create_blocked_dataset(poetry.DATASET_NAME, poetry.poetry_iter("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/poetry/Gutenberg-Poetry.csv"))

def create_quotes_dataset():
    print("qutoes")
    create_blocked_dataset(quotes.DATASET_NAME, quotes.quotes_iter("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/quotes/quotes.csv"))

def create_recipes_dataset():
    print("recipes")
    create_blocked_dataset(recipes.DATASET_NAME, recipes.iter_recipies("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/recipies/recipes_data.csv"))

def create_judicial_dataset():
    print("judicial")
    limited_iterator = islice(judicial.iter_judicial_text("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/judicial/ILDC_multi.csv"), 40_000_000)
    create_blocked_dataset(judicial.DATASET_NAME, limited_iterator)
    
def create_twitter_dataset():
    print("twitter")
    limited_iterator = islice(tweets.tweeter_lines_iter("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/twitter/training_set_tweets.txt"), 40_000_000)
    create_blocked_dataset(tweets.DATASET_NAME, limited_iterator)
    
def create_reddit_dataset():
    print("reddit")
    create_blocked_dataset(reddit.DATASET_NAME, reddit.iter_reddit("/homes/etayl/code/bert/create_dataset/sentence_iterators/sources/reddit/combined_subreddits.csv"))

if __name__ == "__main__":
    create_reviews_dataset()
    create_academics_dataset()
    create_code_dataset()
    create_medical_dataset()
    create_news_dataset()
    create_poetry_dataset()
    create_quotes_dataset()
    create_recipes_dataset()
    create_judicial_dataset()
    create_twitter_dataset()
    create_reddit_dataset()
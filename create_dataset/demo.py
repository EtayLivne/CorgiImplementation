import nltk
import random
from pathlib import Path
# alias set-vastenv="export AWS_ACCESS_KEY_ID=6WK503YYSN97AX83D833 && export AWS_SECRET_ACCESS_KEY=26ZZIZCjUa8D6i+E/4BL9H0MOaW5gbqK0kJqvpE4 && export S3_ENDPOINT=http://vast1.me-corp.lan && export S3_VERIFY_SSL=0 && export S3_USE_HTTPS=0"

# os.environ.update(
#     {
#         "AWS_ACCESS_KEY_ID": "6WK503YYSN97AX83D833",
#         "AWS_SECRET_ACCESS_KEY": "26ZZIZCjUa8D6i+E/4BL9H0MOaW5gbqK0kJqvpE4",
#         "S3_ENDPOINT": "http://vast1.me-corp.lan"
#     }
# )
# def count_book_sentences():
#     books = nltk.corpus.gutenberg.fileids()
#     brown_book_sentences = nltk.corpus.brown.sents(categories=["fiction", "romance", "mystery", "adventure", "humor", "science_fiction", "belles_lettres", "lore", "hobbies"])
#     print(len(brown_book_sentences) + sum(len(nltk.corpus.gutenberg.sents(book)) for book in books))
# print(sum(len(nltk.corpus.gutenberg.sents(book)) for book in books))


# def create_books_dataset():
#     brown_book_sentences = nltk.corpus.brown.sents(categories=["fiction", "romance", "mystery", "adventure", "humor", "science_fiction", "belles_lettres", "lore", "hobbies"])
#     books = nltk.corpus.gutenberg.fileids()
#     gutenberg_sentences = chain(*[nltk.corpus.gutenberg.sents(book) for book in books])
#     all_book_sentences = chain(brown_book_sentences, gutenberg_sentences)
#     create_blocked_dataset(dataset_name="books", sentences_iterator=iter(all_book_sentences), block_size=1000)

# def create_news_dataset():
#     news_files = nltk.corpus.reuters.fileids()
#     news = chain(*[nltk.corpus.reuters.sents(news) for news in news_files])
#     create_blocked_dataset(dataset_name="news", sentences_iterator=iter(news), block_size=1000)
    
# create_news_dataset()
# Download the words dataset from nltk (if not already downloaded)
# nltk.download('words')
nltk.download('gutenberg')
nltk.download('punkt')

# Get a list of English words
word_list = nltk.corpus.words.words()
sentences = nltk.corpus.gutenberg.sents('austen-emma.txt')


def form_a_sentence() -> str:
    return " ".join(random.choice(sentences)) + "\n"

def write_file(where: str, num_sentences: int = 100) -> None:
    with open(where, "w") as handler:
        handler.writelines([form_a_sentence() for _ in range(num_sentences)])


def create_dataset(dirname: str, num_files: int) -> None:
    dirpath = Path(dirname)
    
    for ds_type in ["test", "train", "valid"]:
        ds_dirpath = dirpath / ds_type
        ds_dirpath.mkdir(exist_ok=True, parents=True)
        for i in range(num_files):
            write_file(ds_dirpath/f"{i}.txt")
        
create_dataset("/homes/etayl/code/bert/fake_data", 100)
    
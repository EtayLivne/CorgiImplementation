import nltk
nltk.download('punkt')  # Ensure that the punkt tokenizer is downloaded

from nltk import sent_tokenize

text = "This is a sample text. It contains multiple sentences! How can we split it into sentences? Let's find out."

sentences = sent_tokenize(text)

for sentence in sentences:
    print(sentence)

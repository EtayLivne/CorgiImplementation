from pathlib import Path
from itertools import chain

from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer


def train_tokenizer(output_dir: str):

    files = list(chain.from_iterable(list(str(p) for p in Path(f"/homes/etayl/code/bert/fake_data/{ds_type}").glob("*")) for ds_type in ["test", "train", "valid"]))


    # Initialize a new BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer()

    # Customize tokenizer training parameters (optional)
    tokenizer.train(files=files, vocab_size=30000, min_frequency=2, special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ])

    # Save the trained tokenizer to a file
    tokenizer.save_model(output_dir)
    
def get_pretrained_tokenizer(vocab_path: str):
    return BertTokenizer.from_pretrained(vocab_path)

if __name__ == "__main__":
    train_tokenizer("/homes/etayl/code/bert/bert_tokenizer")
from pathlib import Path
from itertools import chain

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace


# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

files = list(chain.from_iterable(list(str(p) for p in Path(f"/homes/etayl/code/bert/fake_data/{ds_type}").glob("*")) for ds_type in ["test", "train", "valid"]))

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files, trainer)
tokenizer.save_model("dummy_tokenizer.json")

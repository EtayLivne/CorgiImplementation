from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer



tokenizer = BertTokenizer.from_pretrained("/homes/etayl/code/bert/tokenizer/bert_tokenizer/vocab.txt")
# tokenizer = BertTokenizer.from_pretrained("/homes/etayl/code/bert/tokenizer/train_bpe.json")
print(type(tokenizer))

print(tokenizer.encode_plus("shells", "prone"))
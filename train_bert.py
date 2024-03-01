from itertools import chain
from pathlib import Path

from tokenizer.train_bert_wordpiece import get_pretrained_tokenizer
from data.corgi_dataset import LocalTextCorgiDataSet
from data.corgi_sampler import CorgiSampler
from data.tokenized_dataset import TokenizedDataset
from model.litbert import BertForCustomPretraining


from transformers import BertConfig
import lightning.pytorch as pl
import torch


if __name__ == '__main__':
    tokenizer = get_pretrained_tokenizer("/homes/etayl/code/bert/tokenizer/bert_tokenizer/vocab.txt")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )

    # Replace this with your own data loading code
    # Initialize your dataset and dataloader
    files = list(chain.from_iterable(list(str(p) for p in Path(f"/homes/etayl/code/bert/fake_data/{ds_type}").glob("*")) for ds_type in ["test", "train", "valid"]))
    with open(files[0], "r") as handler:
        b = len(handler.readlines())
    corgi_dataset = LocalTextCorgiDataSet(files, b)
    train_dataset = TokenizedDataset(corgi_dataset, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Initialize the LightningModule
    model = BertForCustomPretraining(config)

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=3)  #  gpus=1, Adjust gpus as needed

    # Start training
    trainer.fit(model, train_loader)

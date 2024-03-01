from typing import Iterator
from torch.utils.data import Dataset, IterableDataset
from tokenizers import Tokenizer
import torch
from random import randint

class BertTokenizedDataset(Dataset):
    def __init__(self, other_dataset: Dataset, tokenizer: Tokenizer):
        self._ds = other_dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        item = self._ds[index]
        tokens = self.tokenizer.encode_plus(
            item,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=100, #TODO choose meaningful number
            return_tensors='pt'
        )
        return {
            'input_ids': tokens['input_ids'].flatten(),
            'attention_mask': tokens['attention_mask'].flatten(),
            'labels': tokens['labels'].flatten()
        }
        

class GPT2TokenizedDataset(Dataset):
    def __init__(self, other_dataset: Dataset, tokenizer: Tokenizer, max_length=128):
        self._ds = other_dataset
        self.tokenizer = tokenizer
        self._max_length = max_length
        self._error_rate = 0
        self._iter_rate = 0
    
    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        self._iter_rate += 1
        item = self._ds[index]
        while item is None or len(item) == 0:
            if item is None:
                self._error_rate += 1
            new_index = randint(0, len(self._ds))
            item = self._ds[new_index]
        if self._iter_rate % 2000 == 0:
            print(f"error rate: {self._error_rate/self._iter_rate}")
        return self.tokenizer(item, max_length=self._max_length, padding="max_length", return_tensors="pt", truncation=True)   # truncation=True
    

class GPT2IterableTokenizedDataset(IterableDataset):
    def __init__(self, other_dataset: IterableDataset, tokenizer: Tokenizer, max_length=128):
        self._ds = other_dataset
        self.tokenizer = tokenizer
        self._max_length = max_length
        
    def __iter__(self) -> Iterator:
        for item in self._ds:
            if item is None or len(item) == 0:
                item = "\n"
            yield self.tokenizer(item, max_length=self._max_length, padding="max_length", return_tensors="pt", truncation=True)
    
    # def __len__(self):
    #     return len(self._ds)

    # def __getitem__(self, index):
    #     item = self._ds[index]
    #     return self.tokenizer(item, max_length=self._max_length, padding="max_length", return_tensors="pt", truncation=True)   # truncation=True
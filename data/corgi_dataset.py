from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.dataloader import _BaseDataLoaderIter
import numpy as np
from math import floor
from s3path import S3Path
import boto3
import os 
import random

from .utils import predefined_files_list

# TRAIN_FILES, VAL_FILES = predefined_files_list("file_categories.json")



ENDPOINT_URL = 'http://vast1.me-corp.lan'



class CorgiDataSet(Dataset):
    def __init__(self, files: list[str], b: int, max_cache_size: int=1e6) -> None:
        self.b = b
        self.files = files
        self.num_files = len(self.files)
        self._cache = dict()
        self._iter_count = 0
        self._file_open_count = 0
        self._error_count = 0
        self._file_error_count = []

    
    def _stupid_log(self, log_str):
        with open("dummy_log.log", "a") as handler:
            handler.write(f"{log_str}\n")
    
    def _get_entire_file(self, file_num: int):
        raise NotImplementedError
    
    def _get_indexes_of_items_in_file(self, file_index: int):
        return range(file_index * self.b, (file_index + 1) * self.b)
    
    def _update_cache_with_all_items_in_file(self, file_index: int) -> None:
        item_indexes = self._get_indexes_of_items_in_file(file_index)
        items = self._get_entire_file(file_index)
        if "code" in self.files[file_index]:
            items = [i for i in items if len(i) > 0]
            if len(items) == 0:
                raise ValueError("ZERO PROBLEMS!")
        items = items[:self.b]
        if len(items) < self.b:
            if "code" not in self.files[file_index]:
                print(f"wrong number of elements in file {self.files[file_index]}, num elements: {len(items)}")
            else:
                items = items + random.choices(items, k=(self.b - len(items)))
        
        non_empty_line_indexes = {i for i in range(len(items)) if len(items[i]) > 0}
        num_missing_elements = len(items) - len(non_empty_line_indexes)
        if num_missing_elements > 0:
            non_empty_items = [items[i] for i in range(len(items)) if i in non_empty_line_indexes]
            
            if ("recipes" in self.files[file_index] and len(non_empty_line_indexes)/len(items) < 0.75) or ("recipes" not in self.files[file_index] and len(non_empty_line_indexes)/len(items) < 0.95) :
                items = [i if len(i) > 0 else "naught to do" for i in items]
                if "code" not in self.files[file_index]:
                    self._file_error_count.append((self.files[file_index], len(non_empty_line_indexes)/len(items)))
            
            items = non_empty_items + random.choices(non_empty_items, k=num_missing_elements)
        self._cache.update(dict(zip(item_indexes, items)))
    
    def __len__(self):
        return self.num_files * self.b
    
    def __getitem__(self, index):
        # The cache thing is very efficient assuming data sampled without repetition!
        # if self._iter_count % 1000 == 0 and self._iter_count > 0:
            # print(f"{self._iter_count}: index errors: {self._error_count / self._iter_count}")
            # print(f"{self._iter_count}: file errors: {self._file_error_count}")
        with open(f"log_{get_worker_info().id}.log", "a") as handler:
            # handler.write(f"{self._iter_count}: cache size: {len(self._cache)}")
            handler.write(f"{index}\n")
            # print(f"{self._iter_count}: cache size: {len(self._cache)}")
        index = index if index >= 0 else (len(self) + index)
        if not (index in self._cache):
            self._file_open_count += 1
            file_index = floor((index / self.b))
            self._update_cache_with_all_items_in_file(file_index)
        try:
            self._iter_count += 1
            return self._cache.pop(index, None)

        except Exception as e:
            print(f"got error {e} at index {index}")
            self._error_count += 1
            return "naught to do"
                    

class LocalTextCorgiDataSet(CorgiDataSet):
    def _get_entire_file(self, file_num: int):
        with open(self.files[file_num], "r") as handler:
            sentences = [l.rstrip() for l in handler.readlines()]
        return sentences

class VastTextCorgiDataset(CorgiDataSet):
    @staticmethod
    def infer_b(file):
        session = boto3.session.Session()
        s3 = session.resource(
            "s3", 
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        jpeg_path = file.replace("s3://", "")
        sep = jpeg_path.find("/")
        bucket_name = jpeg_path[:sep]
        key = jpeg_path[sep + 1 :]
        response = s3.Bucket(bucket_name).Object(key).get()
        text = response["Body"].read().decode('utf-8')
        sentences = [l.rstrip() for l in text.split("\n")]
        return len(sentences)
    
    def __init__(self, files: list[str], b: int, max_cache_size: int = 1000000) -> None:
        super().__init__(files, b, max_cache_size)
        self.session = boto3.session.Session()
        
    def _read_txt(self, path: str):
        s3 = self.session.resource(
            "s3", 
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )
        jpeg_path = path.replace("s3://", "")
        sep = jpeg_path.find("/")
        bucket_name = jpeg_path[:sep]
        key = jpeg_path[sep + 1 :]
        
        try:
            response = s3.Bucket(bucket_name).Object(key).get()
            text = response["Body"].read().decode('utf-8')
            if text is None:
                raise ValueError
            return text
        except Exception as e:
            print('error in key:')
            print(key)
            print(e)
            raise e
        
    def _get_entire_file(self, file_num: int):
        text = self._read_txt(self.files[file_num])
        if not isinstance(text, str):
            print(f"ERROR IN: {self.files[file_num]}")
            raise ValueError
        sentences = [l.rstrip() for l in self._read_txt(self.files[file_num]).split("\n")]
        # with open(p, "r") as handler:
        #     sentences = [l.rstrip() for l in handler.readlines()]
        return sentences

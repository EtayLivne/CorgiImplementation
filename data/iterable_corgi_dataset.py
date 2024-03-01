from typing import Iterator
from torch.utils.data import IterableDataset, get_worker_info
import os
import boto3
import random
from itertools import islice, chain
import random


ENDPOINT_URL = 'http://vast1.me-corp.lan'


class IterableCorgiDataset(IterableDataset):
    def __init__(self, files: list[str], files_per_block: int, lines_per_file: int,  output_blocks: bool=False, output_block_size: int=None, local: bool=False) -> None:
        super().__init__()
        self.files = files
        self.files_per_block = files_per_block
        self.output_blocks = output_blocks
        self.output_blocks_size = output_block_size
        self.local = local
        self.lines_per_file = lines_per_file
        
        
    def _read_text_remote(self, path: str) -> str:
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
        
    def _read_text_local(self, path: str) -> str:
        try:
            with open(path, "r") as handler:
                text = handler.read()
            return text
        except Exception as e:
            print(f'error in file: {path}')
            raise e
        
    def _read_text(self, path: str) -> str:
        if self.local is True:
            return self._read_text_local(path)
        else:
            return self._read_text_remote(path)
        
    
    def _get_all_lines_from_file(self, file):
        text = self._read_text(file)
        if not isinstance(text, str):
            print(f"ERROR IN: {file}")
            raise ValueError
        sentences = [l.rstrip() for l in text.split("\n")][:self.lines_per_file]
        if len(sentences) < self.lines_per_file:
            non_empty_lines = [s for s in sentences if len(s) > 0]
            sentences = non_empty_lines + random.choices(non_empty_lines, k=self.lines_per_file-len(non_empty_lines))
            
        return sentences
    
    def _prepare_files(self):
        if get_worker_info() is not None:
                worker_info_id = get_worker_info().id
                num_workers = get_worker_info().num_workers
                self.files = list(self.files[worker_info_id::num_workers])
        random.shuffle(self.files)
                
    def __iter__(self) -> Iterator:
        self._prepare_files()
        print(f"corgi dataset {get_worker_info()}: iterating {len(self.files)} with {self.lines_per_file} lines each")
        self.session = boto3.session.Session()
        i = iter(self.files)
        # remaining_files = len(self.files)
        while True:
            block = list(islice(i, None, self.files_per_block))
            if len(block) < self.files_per_block:
                break
            items = list(chain(*[self._get_all_lines_from_file(f) for f in block]))
            random.shuffle(items)
            if self.output_blocks:
                if len(items) % self.output_blocks_size != 0:
                    print(f'block has {len(items)} items, cant be sliced to blocks of size {self.output_blocks_size}')
                    continue
                j = iter(items)
                for _ in range(self.files_per_block):
                    yield list(islice(j, None, self.output_blocks_size))
                print(f"remaining in iter: {len(list(j))}")
                
            else:
                for item in items:
                    yield item
            # remaining_files = remaining_files - len(block)
            # print(f"remaining files: {remaining_files}")
            
        # for f in self.files:
        #     for line in self._get_all_lines_from_file(f):
        #         yield line
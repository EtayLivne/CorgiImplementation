from typing import Iterator
from torch.utils.data import IterableDataset, get_worker_info, Dataset
import os
import boto3
from math import ceil

ENDPOINT_URL = 'http://vast1.me-corp.lan'


class NonCorgiDataset(Dataset):
    def __init__(self, files: list[str], items_per_file: int) -> None:
        super().__init__()
        self.files = files
        self.items_per_file = items_per_file
        
        self.session = boto3.session.Session() 
    
    def _map_index_to_file_index_and_offset(self, index):
        file_index = ceil(index/self.items_per_file)
        index_in_file = (index -1) % self.items_per_file
        return file_index, index_in_file
        
        
    def _read_txt(self, path: str):
        # if get_worker_info() is not None:
        #     s3_session = self.sessions[get_worker_info().id]
        # else:
        #     s3_session = self.sessions[0]
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
    
    def _get_item(self, file, line_index):
        text = self._read_txt(file)
        if not isinstance(text, str):
            print(f"ERROR IN: {file}")
            raise ValueError
        return text.split("\n")[line_index].lstrip()
        # sentences = [l.rstrip() for l in self._read_txt(file).split("\n")]
        # with open(p, "r") as handler:
        #     sentences = [l.rstrip() for l in handler.readlines()]
        return sentences
    
    def __len__(self):
        return len(self.files) * self.items_per_file
    
    def __getitem__(self, index):
        file_index, offset = self._map_index_to_file_index_and_offset(index)
        if file_index > len(self.files):
            file_index = file_index - 1
        try:
            x = self._get_item(self.files[file_index], offset)
            if x is None:
                y = 7
            return x
        except:
            print(f"couldn't get index {index} because it belongs to file indexed {file_index}, but there are only {len(self.files)} files")
        
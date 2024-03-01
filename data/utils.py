import boto3
import os 
from itertools import chain
import json
import random

ENDPOINT_URL = 'http://vast1.me-corp.lan'

session = boto3.session.Session()
s3 = session.resource(
            "s3", 
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
        )

def list_objects_with_prefix(bucket_name, prefix):
    bucket = s3.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        yield f"s3://{bucket_name}/{obj.key}"
        

def runtime_files_list(categories: list[str]) -> list[str]:
    bucket = "mobileye-team-angie"
    global_prefix = "users/etay/nltk_data"
    category_prefixes = [f"{global_prefix}/{category}/" for category in categories]
    return list(chain(*[list_objects_with_prefix(bucket, cp) for cp in category_prefixes]))

def predefined_files_list(json_path: str, train_val_ratio: float=0.8):
    # global_prefix = "mobileye-team-angie/users/etay/nltk_data"
    with open(json_path, "r") as handler:
        conf = json.load(handler)
    train_data = []
    val_data = []
    for category, num_files in conf.items():
        category_files = ([f"{category}/{i}.txt" for i in range(1, num_files)])
        random.shuffle(category_files)
        split_index = int(train_val_ratio * len(category_files))
        train_data.extend(category_files[:split_index])
        val_data.extend(category_files[split_index:])
        
    return train_data, val_data

import time 

start = time.time()

# CATEGORIES = [
#     "judicial",
#     "medical",
#     "news",
#     "poetry",
#     "quotes",
#     "recipes",
#     "reviews",
#     "tweets"
# ]

# CATEGORIES_THAT_NEED_SOME_LOVE = ["academic", "code", "reddit"]

# FILES = files_list(CATEGORIES)
# end = time.time()
# print(f"DONE IN {end - start} seconds, there are {len(FILES)} files")

# FILES = predefined_files_list("file_categories.json")
# print(FILES[100])
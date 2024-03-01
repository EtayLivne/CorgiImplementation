from corgi_dataset import LocalTextCorgiDataSet
from pathlib import Path
from itertools import chain

files = list(chain.from_iterable(list(str(p) for p in Path(f"/homes/etayl/code/bert/fake_data/{ds_type}").glob("*")) for ds_type in ["test", "train", "valid"]))
with open(files[0], "r") as handler:
    b = len(handler.readlines())
    
ds = LocalTextCorgiDataSet(files, b)

print(ds[0])
print(ds[-1])
print(len(ds) == len(list(ds)))
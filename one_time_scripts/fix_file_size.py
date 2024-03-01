import boto3
import os 
import sys

# # Call the function
# list_objects_with_prefix(bucket_name, prefix)
from pathlib import Path
from itertools import islice

def read_all_sentences_in_directory(directory: Path):
    lines = []
    files = directory.glob("*")
    for f in files:
        with open(f, "r") as handler:
            f_lines = handler.readlines()
        lines.extend(f_lines)
        f.unlink()
    for i in range(len(lines)):
        if lines[i][-1] != "\n":
            lines[i] += "\n" 
    return lines

def write_lines_with_chunk_size(lines, chunk_size, dir_to_write_to: Path):
    lines_iter = iter(lines)
    def batch_iter(iter):
        while True:
            l = list(islice(lines_iter, None, chunk_size))
            if len(l) < chunk_size:
                break
            yield l
    
    counter = 0
    for chunk in batch_iter(lines_iter):
        for i in range(len(chunk)):
            if len(chunk[i]) == 0:
                if i > 0:
                    chunk[i] == chunk[i-1]
                else:
                    chunk[i] == "naught to do"
                    
        text = "".join(chunk)
        if text[-1] == "\n":
            text = text[:-1] + "."
        
        with open(dir_to_write_to / f"{counter}.txt", "w") as handler:
            handler.write(text)
        if counter % 500 == 0:
            print(f"{counter} files written")
        counter += 1
        

def resize_dataset_files(input_directory: Path, output_directory: Path, chunk_size: int):
    output_directory.mkdir(exist_ok=True, parents=True)
    lines = read_all_sentences_in_directory(input_directory)
    write_lines_with_chunk_size(lines, chunk_size, output_directory)

if __name__ == "__main__":
    # input_category = sys.argv[1]
        # output_category = sys.argv[2]
    # new_file_size = int(sys.argv[3])
    input_category = "academic"
    output_category = "academic"
    new_file_size = 1000

    resize_dataset_files(
        Path(f"/homes/etayl/code/bert/local_dataset/baseline/{input_category}"),
        Path(f"/homes/etayl/code/bert/local_dataset/baseline/{input_category}"),
        # Path(f"/homes/etayl/code/bert/local_dataset/upload_cache/{output_category}"),
        new_file_size,
    )


from itertools import chain
from pathlib import Path
import os

import comet_ml
from transformers import GPT2Tokenizer
from tokenizer.train_bert_wordpiece import get_pretrained_tokenizer
from data.corgi_dataset import VastTextCorgiDataset
from data.non_corgi_dataset import NonCorgiDataset
from data.iterable_corgi_dataset import IterableCorgiDataset
from conf import (
    TrainConfig,
    local_test_conf,
    corgi_1_precent_b_1000_n_1350_conf,
    corgi_quarter_precent_b_1000_n_340_conf,
    corgi_1_precent_b_2000_n_675_conf,
    corgi_quarter_precent_b_2000_n_170_conf,
    double_corgi_1_precent_b_1000_n_1350_conf,
    double_corgi_quarter_precent_b_1000_n_340_conf,
    double_corgi_1_precent_b_2000_n_675_conf,
    double_corgi_quarter_precent_b_2000_n_170_conf,
    full_shuffle_conf
)
from data.corgi_sampler import CorgiSampler
from data.tokenized_dataset import GPT2TokenizedDataset, GPT2IterableTokenizedDataset
from model.litgpt import GPT2ForCustomPretraining, GPT2Dataset
import time

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.comet import CometLogger
from s3path import S3Path

from transformers import GPT2Config, GPT2LMHeadModel
import lightning.pytorch as pl
import torch
import torch.utils.data
from  torch.utils.data import SequentialSampler

from data.utils import predefined_files_list



# def get_tokenized_corgi_dataset(files: list[str]):
#     b = 1000
#     corgi_dataset = VastTextCorgiDataset(files, b)
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#     return GPT2TokenizedDataset(corgi_dataset, tokenizer)

def get_tokenized_iterable_corgi_dataset(files: list[str], files_per_block: int, lines_per_file):
    corgi_dataset = IterableCorgiDataset(files, files_per_block, lines_per_file, output_blocks=False, local=False)    #1350
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return GPT2IterableTokenizedDataset(corgi_dataset, tokenizer)
    
    
def get_tokenized_non_corgi_dataset(files: list[str], lines_per_file: int):
    non_corgi_dataset = NonCorgiDataset(files, lines_per_file)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return GPT2TokenizedDataset(non_corgi_dataset, tokenizer)    
    

def dataloaders_from_conf(conf: TrainConfig):
    relative_path_in_container = Path(conf.files_json).name
    train_files, val_files = predefined_files_list(relative_path_in_container)
    print(f"first train file: {train_files[0]}")
    print(f"first val file: {val_files[0]}")
    
    if conf.corgi:
        train_dataset = get_tokenized_iterable_corgi_dataset(train_files, conf.files_per_block_train, conf.lines_per_file)
        val_dataset = get_tokenized_non_corgi_dataset(val_files, conf.lines_per_file)
        train_dataloader_shuffle = None
        val_data_loader_shuffle = True
        print("CORGI")
    else:
        train_dataset = get_tokenized_non_corgi_dataset(train_files, conf.lines_per_file)
        val_dataset = get_tokenized_non_corgi_dataset(val_files, conf.lines_per_file)
        train_dataloader_shuffle = True
        val_data_loader_shuffle = True
        print("NON CORGI")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_train_workers,
        prefetch_factor=conf.prefetch_factor,
        persistent_workers=True,
        shuffle=train_dataloader_shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_val_workers,
        prefetch_factor=conf.prefetch_factor,
        persistent_workers=True,
        shuffle=val_data_loader_shuffle
    )

    return train_dataloader, val_dataloader

def get_model_from_conf(conf: TrainConfig):
    gpt_config = GPT2Config()
    gpt_model = GPT2LMHeadModel(gpt_config)

    litmodel = GPT2ForCustomPretraining(
            gpt_model=gpt_model,
        )
    return litmodel


def get_loggers():
    os.environ["COMET_API_KEY"] = "vRiwL5vDZdkgbiXmyG8HlzHYT"
    os.environ["COMET_WORKSPACE"] = "etayl"
    os.environ["COMET_URL_OVERRIDE"] = "http://comet.angie.mobileye.com/clientlib/"
    loggers = [
        CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"), 
            project_name="gpt2", 
            save_dir="./boatlogs"
        )
    ]
    return loggers

def get_callbacks():
    
    callbacks = [
        ModelCheckpoint(
            dirpath="test_checkpoints",
            every_n_train_steps=int(10**3)*5,
            save_top_k=-1,
        ),
        LearningRateMonitor(log_momentum="step")
    ]
    return callbacks
    
def test_iterate_loader(loader):
    print("FART")
    for i, x in enumerate(loader):
        print(i)
        if i > 100:
             print(x)
             break
    exit()

def start_training():
    confs = [
        local_test_conf, full_shuffle_conf, corgi_1_precent_b_1000_n_1350_conf, corgi_quarter_precent_b_1000_n_340_conf,
        corgi_1_precent_b_2000_n_675_conf, corgi_quarter_precent_b_2000_n_170_conf,
        double_corgi_1_precent_b_1000_n_1350_conf, double_corgi_quarter_precent_b_1000_n_340_conf,
        double_corgi_1_precent_b_2000_n_675_conf, double_corgi_quarter_precent_b_2000_n_170_conf
    ]
    
    conf = double_corgi_quarter_precent_b_2000_n_170_conf
    
    train_dataloader, val_dataloader = dataloaders_from_conf(conf)
    
    litmodel = get_model_from_conf(conf)
    loggers = get_loggers()
    callbacks = get_callbacks()
    
    print(dict(conf))
    trainer = pl.Trainer(max_steps=conf.max_steps, log_every_n_steps=5, val_check_interval =conf.train_steps_between_vals, limit_val_batches=conf.val_steps, callbacks=callbacks, logger=loggers, num_sanity_val_steps=0)
    trainer.fit(litmodel, train_dataloader, val_dataloader)

    
if __name__ == '__main__':
    start_training()    
     #save_dir="",  


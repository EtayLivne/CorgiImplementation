# FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.13-cuda12.0.1
# RUN pip install einops==0.5.0 torch==1.13.1 'mosaicml[nlp,wandb]>=0.14.0,<0.15' mosaicml-streaming==0.4.1 omegaconf==2.2.3 transformers==4.28.1 triton==2.0.0.dev20221103
FROM python:latest
RUN pip install s3path
RUN pip install torch
RUN pip install tokenizers
RUN pip install transformers
RUN pip install lightning
RUN pip install boto3
RUN pip install comet-ml
RUN pip install pydantic
WORKDIR /app
COPY train_gpt.py /app
COPY tokenizer /app/tokenizer
COPY model /app/model
COPY data /app/data
COPY file_categories.json /app
COPY single_corgi_categories.json /app
COPY file_categories_b_2000.json /app
COPY files_corgi_b_2000_n_170.json /app
COPY files_corgi_b_2000_n_675.json /app
COPY conf.py /app
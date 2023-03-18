#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:32:57 2023

@author: aus79
"""

import os
from argparse import ArgumentParser
import numpy as np
import random
import pickle
import tensorflow as tf

from tqdm import tqdm
from typing import List
from multiprocessing import Process, Manager, Queue
from pathlib import Path
from customtokenizer import tokenize
from trainmed import prepare,train
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2Tokenizer
from transformers import WEIGHTS_NAME, CONFIG_NAME
def parse_args():
    """Command-line argument parser for training."""

    # New parser
    parser = ArgumentParser(description='GPT2 tensorflow model for custom medical text data')

    # Data parameters
    parser.add_argument('-wd', '--working-dir', help='path to raw text', default='/datasets/')
    parser.add_argument('-t', '--raw-text', help='path to raw text', default='./datasets/raw.txt')
    parser.add_argument('-sp', '--split-text', help='number of text splits', default=1, type=int)
    parser.add_argument('-gz', '--gpt-size', help='size of GPT2 model. [small / medium / large]', default='small', type=str)
    parser.add_argument('-tp', '--tokenizer-path', help='path to save tokenizer', default='./datasets/tokenz/')
    parser.add_argument('-m', '--model-path', help='path to save model', default='./datasets/models/')
    parser.add_argument('-ct', '--raw-cut-path', help='path to raw cut text', default='train_data/')
    
    parser.add_argument('-b', '--batch-size', help='minibatch size', default=12, type=int)
    parser.add_argument('-blk', '--block-size', help='size of each text block', default=100, type=int)
    parser.add_argument('-bff', '--buffer-size', help='buffer size', default=1000, type=int)
    
    parser.add_argument('-lr', '--learning-rate', help='learning  rate', default=0.0001, type=float)
    parser.add_argument('-e', '--nb-epochs', help='number of epochs', default=1, type=int)
    parser.add_argument('-ep', '--epsilon', help='adam parameter epsilon', default=1e-8, type=float)
    
    parser.add_argument('-tsp', '--text_sample', help='sample text for generation', default="what is", type=str)
    
    return parser.parse_args()



def cut_words(processer_num, text, result_dict):
    current_directory = os.getcwd()
    train_path=os.path.join(current_directory, params.raw_cut_path)
    #train_path=params.raw_cut_path
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    with open(os.path.join(train_path, f'raw.cut.temp.{processer_num}.txt'), 'w') as out_f:
        texts = text.split('\n')
        for line in tqdm(texts):
            try:
                #cuts = " ".join(jieba.cut(line))
                cuts=" ".join(line.split())
                out_f.write(cuts+'\n')
            except UnicodeDecodeError:
                pass
            except KeyError:
                pass
            except Exception as e:
                pass


def multiply_cut(handler, tasks):
    manager = Manager()
    result_dict = manager.dict()  # didn't work and don't know why
    jobs = []
    for processer_num, task in enumerate(tasks):
        p = Process(target=handler, args=(
            processer_num, task, result_dict))
        jobs.append(p)

    for job in jobs:
        job.start()

    for job in jobs:
        job.join()

    for job in jobs:
        try:
            job.close()  # It may raise exception in python <=3.6
        except:
            pass
    print("[all_task done]")


def split_data(text,params):
    text_task = []
    num_pre_task = len(text)//params.split_text
    for i in range(0, len(text), num_pre_task):
        text_task.append(text[i: i + num_pre_task])
    return text_task
def preprocess(params,train_path):
    
    print(f'reading {params.raw_text}')
    with open(params.raw_text, 'r') as f:
        data = f.read().replace('  ', ' ').replace('\n\n', '\n')
        print(f"total words: {len(data)}") #352124

    print(f"split data into {params.split_text} pieces")
    text_task = split_data(data,params)

    multiply_cut(cut_words, text_task)
    cut_path=train_path+'raw_text_cut.txt'
    path = Path(train_path)
    with open(cut_path, 'w') as all_cut_file:
        for filename in path.glob('raw.cut.temp.*'):
            with open(filename) as cut_file:
                all_cut_file.write(cut_file.read()+'\n')
                print(f'dropping {filename}')
                os.system(f'rm {filename}')


if __name__ == '__main__':
    params=parse_args()
    current_directory = os.getcwd()
    train_path=os.path.join(current_directory, params.raw_cut_path)
    #train_path=params.raw_cut_path
    if not os.path.exists(train_path):
        os.mkdir(train_path)
        
    preprocess(params,train_path)
    print("data preprocess complete")
    _=tokenize(train_path,params.tokenizer_path)
    print("tokenizer training complete")
    model,tokenizer,string_tokenized=prepare(params)
    print("model architecture and data prep complete")
    model,tokenizer=train(params,model,tokenizer,string_tokenized)
    print(" model training complete")
    
    output_dir = params.model_path# creating directory if it is not present
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)# save model and model configs
    model.save_pretrained(output_dir)
    model_to_save.config.to_json_file(output_config_file)# save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    model = TFGPT2LMHeadModel.from_pretrained(output_dir)
    sampletext=params.text_sample
    input_ids = tokenizer.encode(sampletext, return_tensors='tf')# getting out output
    beam_output = model.generate(
      input_ids,
      max_length = 128,
      num_beams = 5,
      temperature = 0.8,
      no_repeat_ngram_size=2,
      num_return_sequences=5
    )
    outp=tokenizer.decode(beam_output[4]).replace('<|eos|>','')
    print(outp)
    

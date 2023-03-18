#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:03:09 2023

@author: aus79
"""


import os
from transformers import GPT2TokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
def tokenize(pathtorawcut,pathtosave):
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.pre_tokenizer.pre_tokenize_str("Let's test pre-tokenization!")
    trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<|bos|>","<|eos|>","<|pad|>"])
    files=[pathtorawcut+x for x in os.listdir(pathtorawcut)]
    tokenizer.train(files=files,trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    
    
    wrapped_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer,bos_token="<|bos|>",eos_token="<|eos|>",unk_token="<|unk|>")
    wrapped_tokenizer.save_pretrained(pathtosave)
    return wrapped_tokenizer
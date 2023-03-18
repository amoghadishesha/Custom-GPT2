
import os
import tensorflow as tf
from transformers import GPT2Config, TFGPT2LMHeadModel, GPT2TokenizerFast,GPT2Tokenizer# loading tokenizer from the saved model path
import numpy as np
from transformers import WEIGHTS_NAME, CONFIG_NAME
import os
#tokenizer = GPT2Tokenizer.from_pretrained('./bpenew/')

def getarch(sz):
    if sz=='small':
        n_layer=12
        n_head=12
        n_embd= 768
    elif sz=='medium':
        n_layer=24
        n_head=16
        n_embd= 1024
    else:
        n_layer=36
        n_head=20
        n_embd= 1280
    return n_layer,n_head,n_embd


def prepare(params):
    vocabfile=params.tokenizer_path+'vocab.json'
    mergefile=params.tokenizer_path+'merges.txt'
    tokenizerfl=params.tokenizer_path+'tokenizer.json'
    size=params.gpt_size
    n_layer,n_head,n_embd=getarch(size)
    tokenizer = GPT2Tokenizer.from_pretrained(params.tokenizer_path)
    #tokenizer=GPT2TokenizerFast(vocab_file=vocabfile, merges_file=mergefile, tokenizer_file=tokenizerfl,n_layer=n_layer,n_head=n_head,n_embd=n_embd)

    config = GPT2Config(vocab_size=tokenizer.vocab_size, bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id)
    
    model = TFGPT2LMHeadModel(config)
#model = TFGPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path='./bpenew/')
    current_directory = os.getcwd()
    train_path=os.path.join(current_directory, params.raw_cut_path)
    paths=[train_path+x for x in os.listdir(train_path)]
    #paths=['./dataset/train/raw.cut.txt']
    encodedstring=[]
    for filename in paths:
      with open(filename, "r", encoding='utf-8') as f:
          lines=f.readlines()
          for line in lines:
              x= tokenizer.bos_token+line + tokenizer.eos_token
              tox=tokenizer.encode(x)     
              if None not in tox:
                  encodedstring.append(tox)
       #single_string += x + tokenizer.eos_token
    #string_tokenized = tokenizer.encode(single_string)
    string_tokenized = list(np.concatenate(encodedstring).flat)
    return model,tokenizer,string_tokenized

def train(params,model,tokenizer,data):
    string_tokenized=data
    examples = []
    block_size = params.block_size
    
    BATCH_SIZE = params.batch_size
    if params.gpt_size=='large':
        if BATCH_SIZE>2:
            BATCH_SIZE=2
    BUFFER_SIZE = params.buffer_size
    for i in range(0, len(string_tokenized) - block_size + 1, block_size):
      examples.append(string_tokenized[i:i + block_size])
    inputs, labels = [], []
    for ex in examples:
      inputs.append(ex[:-1])
      labels.append(ex[1:])
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # defining our optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)# definining our loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)# defining our metric which we want to observe
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])

    num_epoch = params.nb_epochs
    history = model.fit(dataset, epochs=num_epoch)
    
    output_dir = params.model_path
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    model.save_pretrained(output_dir)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_pretrained(output_dir)
    return model,tokenizer

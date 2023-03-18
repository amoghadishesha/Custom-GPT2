# Custom-GPT2

## General comments:
This is edited from [mymusise/gpt2-quickly](https://github.com/mymusise/gpt2-quickly#main-file) . 
The main changes include a BPE tokenizer instead of the sentence-piece (works better for English on GPT2) and a custom tokenizer wrapper
Additionally, the data preprocessing and storage are heavily edited for simplicty and to match english language
finally, the code includes a simple to use terminal interface along with an option to use a model I trained for 100 epochs

## Usage:
Please install all requirements in requirements.txt using the following 

* pip install -r requirements.txt

If you are using an interpreter or other visual interfaces, open main.py and update the arguments according to your preferred folder structure
If you are using a terminal interace use the follwing sample code to run the training process-


* python main.py - --working-dir='./gpt/' --raw-text='./gpt/dataset/raw.txt/ -tr=True --nb-epochs=100 

 The folder structure is as follows:
- Working Directorty
  - datasets
    - raw.txt
   - trainng_data
   - models

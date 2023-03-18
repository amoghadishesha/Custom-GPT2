# Custom-GPT2

## General comments:
This is edited from [mymusise/gpt2-quickly](https://github.com/mymusise/gpt2-quickly#main-file) . 
The main changes include a BPE tokenizer instead of the sentence-piece (works better for English on GPT2) and a custom tokenizer wrapper
Additionally, the data preprocessing and storage are heavily edited for simplicty and to match english language.
Finally, the code includes a simple to use terminal interface along with an option to use a model I trained for 100 epochs

## Usage:
Please install all requirements in requirements.txt using the following 

* pip install -r requirements.txt

## Run the code (training and Prediction)
* python main.py

## To run with trained weights:
1. Download weights from google drive (check trained_weights readme) and save it in the the trained_weights folder
2. Then run :
 * python main.py -wtr 

 The folder structure is as follows:
- Working Directorty
  - datasets
    - raw.txt
   - trainng_data
   - models

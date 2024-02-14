import re
import torch
from datasets import load_dataset
import pandas as pd
from transformers import TrainingArguments, Trainer, GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig

PARAMS = '14M'
config = pd.read_csv('pythia.csv')
config = config[config['params'] == PARAMS]

# creating model with experiment configuration
configuration = GPTNeoXConfig()
configuration.intermediate_size = int(config['intermediate_size'].values[0])
configuration.num_attention_heads = int(config['n_heads'].values[0])
configuration.num_hidden_layers =  int(config['n_layers'].values[0])
configuration.layer_norm_eps= float(config['Learning Rate'].values[0])
configuration.vocab_size = 8
model = GPTNeoXForCausalLM(configuration)



model = GPTNeoXForCausalLM.from_config(
  configuration
)

tokenizer = AutoTokenizer.from_pretrained(
  "/content/virus_pythia_small/checkpoint-30500"
)

ds = load_dataset('Hack90/virus_10kbp_50kbp')
import re
import torch
from datasets import load_dataset
import pandas as pd
from transformers import TrainingArguments, Trainer, GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, AutoTokenizer, GPTNeoXConfig

PARAMS = '14M'
config = pd.read_csv('/content/pythia.csv')
config = config[config['Params'] == PARAMS]

# creating model with experiment configuration
configuration = GPTNeoXConfig()
configuration.intermediate_size = int(config['intermediate_size'].values[0])
configuration.num_attention_heads = int(config['n_heads'].values[0])
configuration.num_hidden_layers =  int(config['n_layers'].values[0])
configuration.layer_norm_eps= float(config['Learning Rate'].values[0])
configuration.vocab_size = 8
model = GPTNeoXForCausalLM(configuration)

tokenizer = AutoTokenizer.from_pretrained(
  "hack90/virus_pythia_small"
)

ds = load_dataset('Hack90/virus_10kbp_50kbp')

def remove_every_token_not_in_atcg(examples):
  text = examples['sequence']
  text = re.sub(r'[^atcg]', 'n', text)
  return {'text':text}

ds = ds.map(remove_every_token_not_in_atcg)

# Function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=config['max_pos_emb'].values[0])

# Tokenize all splits
tokenized_datasets = ds.map(tokenize_function, batched=True)

# Format the dataset to output PyTorch tensors
tokenized_datasets.set_format(type='torch')
tokenizer.pad_token = tokenizer.eos_token

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss

# Defining the training arguments
training_args = TrainingArguments(
    output_dir=f"./virus_pythia_{PARAMS}",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Initializing the Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset= tokenized_datasets['valid'],
    tokenizer=tokenizer,
)

# Training the model
trainer.train()
## Script to train a BPETokenizer from transformers on the training data.
## The tokenizer is saved to huggingface hub.

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer
from transformers import AutoTokenizer

vs_ds = load_dataset("Hack90/virus_bert_chunk_2kbp")
# remove whitespace from sequences and chunk into 7 kmers
vs_ds = vs_ds.map(lambda x: {"sequence": x["sequence"].replace(" ", "")})
vs_ds = vs_ds.map(lambda x: {"sequence": [x["sequence"][i:i+7] for i in range(0, len(x["sequence"]), 7)]})
vs_ds = vs_ds.map(lambda x: {"sequence": [i for i in x["sequence"] if len(i) == 7]})
vs_ds = vs_ds.map(lambda x: {"sequence": " ".join(x["sequence"])})

tokenizer = ByteLevelBPETokenizer()

tokenizer.train_from_iterator(vs_ds["train"]["sequence"], vocab_size=1000, min_frequency=2)
tokenizer.save_model("Hack90/virus_bert_chunk_2kbp_tokenizer")

#tokenize dataset,pad and save to huggingface hub
tokenized_dataset = vs_ds.map(lambda x: tokenizer.encode(x["sequence"], truncation=True, padding="max_length", max_length=1000), batched=True)
tokenized_dataset.push_to_hub("Hack90/virus_bert_chunk_2kbp_tokenized")





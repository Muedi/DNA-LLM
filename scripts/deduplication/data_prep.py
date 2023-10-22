from datasets import load_dataset, concatenate_datasets
from itertools import islice

## set your huggingface username
username = "Hack90"

## reshape sequences to chunks of spaced 2000 nucleotides
def chunk_sequences(examples: dict) -> dict:
    chunks = []
    chunk_length = []
    for batch in batched(list(examples['sequence_x']), 2000):
        chunks.append(' '.join(batch))
        chunk_length.append(len(batch))
    return {'chunks': chunks, 'chunk_length': chunk_length}


def batched(iterable: list, n: int) -> list:
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

## load the partial datasets and save them to a list
datasets = []
for i in range(0, 78):
    try:
        dataset = load_dataset(f"Hack90/ncbi_genbank_part_{i}")
        datasets.append(dataset)
    except(FileNotFoundError):
        print(f"Failed to load dataset {i}")

## concatenate the partial datasets to one full dataset
full_dataset = concatenate_datasets([datasets])

## filter the dataset to only include sequences with length 2000
full_dataset = full_dataset.filter(lambda example: example["seq_length"] == 2000)

## reshape sequences to chunks of spaced 2000 nucleotides
full_dataset = dataset.map(chunk_sequences)

## save the dataset to local disk and push it to huggingface dataset
full_dataset.save_to_disk("dna_llm/ncbi_genbank_full_2000_bp_chunks")
full_dataset.push_to_hub(f"{username}/ncbi_genbank_full_2000_bp_chunks")









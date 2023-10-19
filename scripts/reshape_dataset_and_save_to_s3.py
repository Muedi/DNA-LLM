from datasets import load_dataset
from datasets import concatenate_datasets

## load the partial datasets and concatenate them
for i in range(0, 78):
    try:
        dataset = load_dataset(f"Hack90/ncbi_genbank_part_{i}")
    except:
        print(f"Failed to load dataset {i}")
        continue
    if i == 1:
        full_dataset = dataset
    else:
        full_dataset = concatenate_datasets([full_dataset, dataset])

## save the full dataset to s3
full_dataset.save_to_disk("s3://dna_llm/ncbi_genbank_full")

## filter the dataset to only include sequences with length 2000
full_dataset = full_dataset.filter(lambda example: example["seq_length"] == 2000)

## reshape sequences to batches of 2000
full_dataset = full_dataset.map(lambda example: {"sequence": [example["sequence"][i:i+2000] for i in range(0, len(example["sequence"]), 2000)]})

full_dataset.save_to_disk("s3://dna_llm/ncbi_genbank_full_2000")







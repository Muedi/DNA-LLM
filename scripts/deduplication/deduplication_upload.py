from datasets import load_from_disk

# set your huggingface username
USERNAME = "Hack90"
# Load the dataset from disk
dataset = load_from_disk('output/minhash/ncbi_genbank_full_2000_bp_chunks/')

# Upload to the huggingface datasets hub
dataset.push_to_hub(f'{USERNAME}/ncbi_genbank_deduplicated_2000_bp_chunks')

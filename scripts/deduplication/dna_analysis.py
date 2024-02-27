from datasets import load_dataset
from collections import Counter
import pandas as pd

# Load dataset
ds = load_dataset('Hack90/ref_seq_viral')

# Function to generate k-mers for each sequence
def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Function to get most common k-mers across all sequences in the dataset
def most_common_kmers(k, sequences):
    all_kmers = (kmer for seq in sequences for kmer in generate_kmers(seq.lower(), k))
    return Counter(all_kmers).most_common(20)

# Consolidate sequence data
sequences = ds['train']['sequence']

# Get most common k-mers for k=1 to 19
most_common_kmers_by_k = {k: most_common_kmers(k, sequences) for k in range(1, 20)}

# Create a single dataframe from the collected k-mer counts
df_with_n = pd.concat([
    pd.DataFrame(data=kmers, columns=['kmer', 'count']).assign(k=k)
    for k, kmers in most_common_kmers_by_k.items()
], ignore_index=True)



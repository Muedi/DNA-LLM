from datasets import load_dataset
from collections import Counter
import pandas as pd

from datasets import load_dataset
from collections import Counter
# Load dataset
ds = load_dataset('Hack90/ref_seq_viral')['train']['sequence']

# Efficiently generate all k-mers for a given k in all sequences
def generate_all_kmers(sequences, k):
    return (seq[i:i+k] for seq in sequences for i in range(len(seq) - k + 1))

# Efficient computation of most common k-mers for a range of k values
def compute_most_common_kmers(sequences, max_k, top_n=10000000):
    # Initialize a DataFrame to store results
    df = pd.DataFrame()
    for k in range(1, max_k + 1):
        kmers = generate_all_kmers(sequences, k)
        most_common = Counter(kmers).most_common(top_n)
        temp_df = pd.DataFrame(most_common, columns=['kmer', 'count'])
        # percent of total number of k-mers
        temp_df['percent'] = temp_df['count'] / temp_df['count'].sum()
        temp_df['k'] = k
        df = pd.concat([df, temp_df], ignore_index=True)
        print(df)
    return df

# Convert sequences to lowercase
sequences_lower = [seq.lower() for seq in ds]

# Compute and store results
df_result = compute_most_common_kmers(sequences_lower, max_k=10)

# Display or use the DataFrame as needed
print(df_result)



########################################################################## shiny app ########################################
from shiny import render
from shiny.express import input, ui
import matplotlib.pyplot as plt

ui.page_opts(fillable=True)
ui.panel_title("Kmer Analysis")
with ui.layout_columns():
    with ui.card():
        ui.input_slider("slider", "Slider", 0, 10, 5)

@render.plot
def plot():  
    df = pd.read_csv('kmers.csv')
    k = input.slider()
    fig = None
    if k:
        df = df[df['k'] == k]
        fig, ax = plt.subplots()
        ax.bar(df['kmer'], df['count'])
        ax.set_title(f"Most common {k}-mers")
        ax.set_xlabel("K-mer")
        ax.set_ylabel("Count")
        ax.set_xticklabels(df['kmer'], rotation=90)
    return fig

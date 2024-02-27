import pandas as pd

## read every sheet except overview
dfs ={}
organism_type = ['bacteria', 'mammals', 'plants', 'viral', 'fungi', 'invertebrate',
                'vertebrate_other', 'archea', 'protozoa', 'plasmids', 'plastids', 'mitochondrion']
for organism in organism_type:
    dfs[organism] = pd.read_excel('/workspaces/DNA-LLM/scripts/deduplication/RefSeq.xlsx', sheet_name=organism)
    print(f"{organism} sheet read")

## sum the total number of sequences for each organism
token_sums = {}
for organism in organism_type:
    dfs[organism] = dfs[organism][dfs[organism]['Name'].str.contains('genomic.fna.gz')]
    #sum only numeric values which maybe floats
    tokens = dfs[organism]['Tokens(M)'].apply(lambda x: pd.to_numeric(x, errors='coerce')).sum()
    token_sums[organism] = tokens
    print(f"{organism} token sum: {token_sums[organism]}")

size_sums = {}
for organism in organism_type:
    dfs[organism] = dfs[organism][dfs[organism]['Name'].str.contains('genomic.fna.gz')]
    size = dfs[organism]['Size(M)'].apply(lambda x: pd.to_numeric(x, errors='coerce')).sum()
    size_sums[organism] = size
    print(f"{organism} size sum: {size_sums[organism]}")

## sum the total number of sequences for each organism
total_tokens = sum(token_sums.values())
total_size = sum(size_sums.values())
print(f"Total tokens: {total_tokens}")
print(f"Total size: {total_size}")

## get the gz links for each organism, download and process the files
import time
import subprocess
import concurrent.futures
import logging
import glob
import ftplib
import requests
from bs4 import BeautifulSoup
from itertools import islice
import pandas as pd
from Bio import SeqIO
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def gunzip_file(file):
    """Function to decompress a single file and delete original."""
    logging.info(f"gunzipping {file}")
    subprocess.run(f"gunzip {file}", shell=True)
    logging.info(f"deleting {file}")
    subprocess.run(f"rm {file}", shell=True)

def process_file(file, file_count):
    logging.info(f"processing {file}")
    seqs_data = {
        'id': [],
        'sequence': [],
        'name': [],
        'description': [],
        'features': [],
        'seq_length': []
    }
    try:
        for record in SeqIO.parse(file, "genbank"):
            seqs_data['id'].append(str(record.id))
            seqs_data['sequence'].append(str(record.seq))
            seqs_data['name'].append(str(record.name))
            seqs_data['description'].append(str(record.description))
            seqs_data['features'].append(len(record.features))
            seqs_data['seq_length'].append(len(str(record.seq)))       
        count = file_count[file]  
        df = pd.DataFrame(seqs_data)
        ## checking if any seq length is above the 2GB limit of pyarrow
        ## and if so, splitting that row into batches of 2GB
        if any(df.seq_length > 2_000_000_000):
            n = len(df[df.seq_length > 2_000_000_000])
            logging.info(f"the file has {n} rows with seq_length > 2GB")
            df_less = df[df.seq_length < 2_000_000_000].copy()
            df_more = df[df.seq_length > 2_000_000_000].copy()
            # deleting the original df to save memory
            del df
            # splitting the sequence into batches of 2GB
            df_more['sequence'] = df_more.sequence.apply(lambda x: list(batched(x, 2_000_000_000)))
            # batch numbers for each row
            df_more['batch'] = df_more.sequence.apply(lambda x: len(x))
            # batch id list for each row
            df_more['batch_id'] = df_more.batch.apply(lambda x: list(range(x)))
            # exploding the sequence column and creating a new row for each batch
            df_more = df_more.explode(['sequence','batch_id'])
            # merge sequences
            df_more.sequence = df_more.sequence.apply(lambda x: ''.join(x))
            df_more['seq_length'] = df_more.sequence.str.len()
            # dropping the original df_more row
            df_more = df_more.drop(df_more[df_more.seq_length > 2_000_000_000].index)
            # concat the two dataframes
            df = pd.concat([df_less,df_more])
        logging.info(f"writing {file}")
        df.to_parquet(f'file_{count}.parquet')
    except Exception as e:
        logging.error(f'Error processing {file}: {e}')
    finally:
        subprocess.run(f"rm {file}", shell=True)


def download_file(file, organism):
    command = f'wget https://ftp.ncbi.nlm.nih.gov/refseq/release/{organism}/{file}'
    subprocess.run(command, shell=True)


def process_total(files, organism_type):
    try:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(download_file, files, [organism_type]*len(files))

        gz_files = glob.glob("**.gz")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(gunzip_file, gz_files)

        files = glob.glob("*.seq")
        file_count = {f: idx for idx, f in enumerate(files)}

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_file, files, [file_count]*len(files)))

        for result in results:
            logging.info(result)
    except Exception as e:
        logging.error(f'Error in process_total: {e}')
    finally:
        print('loading parquet files')
        dataset = load_dataset("parquet", data_files="**.parquet")
        date = time.strftime("%Y_%m_%d")
        print('pushing to hub')
        dataset.push_to_hub(f'Hack90/ncbi_genbank_full_v2_{date}')
        dataset.cleanup_cache_files()
        print('done')
        logging.info('Done with process_total')

## remove bacteria, mammals, vertebrate_other from the list
organism_type = ['plants', 'viral', 'fungi', 'invertebrate', 'archea', 'protozoa', 'plasmids', 'plastids', 'mitochondrion']

for organism in organism_type:
    process_total(dfs[organism]['Name'], organism)



                 

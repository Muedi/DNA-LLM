### De-duplicating NCBI Genbank Data

1. Copy this repo to your local enviroment & unzip:
```bash
wget https://github.com/hssn-20/DNA-LLM/archive/refs/heads/main.zip
unzip main.zip
```

2. Install the conda enviroment by running:
```bash
conda env create --file ~DNA-LLM/scripts/deduplication/environment.yml
```

3. Activate the enviroment 
```bash
conda activate deduplication
```
4. Login to Huggingface
```bash
huggingface-cli login
```

5. Run the dataset prep script - before running the data prep script make sure to change the  huggingface username to yours at the start of the script
```bash
python ~DNA-LLM/scripts/deduplication/data_prep.py
```

6. Copy Chenghao Mou's dediplication repo into your enviroment
```bash
wget https://github.com/ChenghaoMou/text-dedup/archive/refs/heads/main.zip -O dedup.zip
unzip dedup.zip
```

7. Run the dedup script:
```bash 
cd text-dedup-main && python -m text_dedup.minhash \
--path "{username}/ncbi_genbank_full_2000_bp_chunks" \
--split "train" \
--cache_dir "./cache" \
--output "output/minhash/ncbi_genbank_full_2000_bp_chunks" \
--column "chunks" \
--batch_size 10000 \
--num_perm 250 \
--ngram 2
```

8. Upload the data to hugginface - once again remember to set your huggingface username:
```bash 
python scripts/deduplication/deduplication_upload.py
```

The whole process will take a few hours to run.


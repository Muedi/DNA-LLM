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

5. Run the dataset prep script - before running the data prep script make sure to fix the huggingface
```bash
python ~DNA-LLM/scripts/deduplication/data_prep.py
```



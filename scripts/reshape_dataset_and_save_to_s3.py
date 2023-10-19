import boto3
from datasets import load_dataset_builder

# Create an S3 client
s3_client = boto3.client('s3')

# Set up your storage options
# Note: You may not need this depending on your subsequent code.
storage_options = {"client": s3_client}

# Rest of your code remains similar...
builder = load_dataset_builder("Hack90/virus_dna_dataset")
output_dir = "s3://dna-llm/"
builder.download_and_prepare(output_dir, storage_options=storage_options, file_format="parquet")

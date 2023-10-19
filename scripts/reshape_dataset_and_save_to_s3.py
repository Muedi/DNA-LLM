from botocore.session import Session
import s3fs
from datasets import load_dataset_builder
storage_options = {"key": "XXX",

"secret": "XXX"}
s3_session = Session(profile="your_profile_name")

storage_options = {"session": s3_session}
fs = s3fs.S3FileSystem(**storage_options)
builder = load_dataset_builder("your dataset id")
output_dir = "s3://path/to/the/bucket/"
builder.download_and_prepare(output_dir, storage_options=storage_options, file_format="parquet")
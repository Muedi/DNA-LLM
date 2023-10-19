python -m text_dedup_main/text_dedup.minhash \
  --path "Hack90/ncbi_genbank_full_2000_bp_chunks" \
  --split "train" \
  --cache_dir "./cache" \
  --output "output/minhash/ncbi_genbank_full_2000_bp_chunks" \
  --column "chunks" \
  --batch_size 10000

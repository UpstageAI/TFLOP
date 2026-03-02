data_config_path="dataset/data_preprocessing_config.json"
output_dir="TFLOP-dataset/meta_data"

# Original: runs all data at once (may cause OOM / SSH disconnection on large datasets)
# uv run -m dataset.preprocess_data --data_config_path $data_config_path \
#                                   --output_dir $output_dir

# Process in bins to reduce peak memory usage
num_bins=8

for bin_idx in $(seq 0 $((num_bins - 1))); do
    echo "Processing bin ${bin_idx}/$((num_bins - 1))..."
    uv run -m dataset.preprocess_data --data_config_path $data_config_path \
                                      --output_dir $output_dir \
                                      --bin_idx $bin_idx \
                                      --num_bins $num_bins
done

# Merge bin files into the single file expected by the training code
for split in train validation; do
    echo "Merging ${split} bins..."
    cat ${output_dir}/dataset_${split}_*_${num_bins}.jsonl > ${output_dir}/dataset_${split}.jsonl
    echo "Saved to ${output_dir}/dataset_${split}.jsonl"
done
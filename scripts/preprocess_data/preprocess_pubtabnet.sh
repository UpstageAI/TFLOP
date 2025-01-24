data_config_path="dataset/data_preprocessing_config.json"
output_dir="TFLOP-dataset/meta_data"

python -m dataset.preprocess_data --data_config_path $data_config_path \
                                  --output_dir $output_dir

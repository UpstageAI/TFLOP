bin_idx=$1
total_bin_cnt=$2
exp_savepath=$3
epoch_step_checkpoint=$4

pubtabnet_aux_json_path="TFLOP-dataset/meta_data/final_eval_v2.json"
pubtabnet_aux_img_path="TFLOP-dataset/images/test"
pubtabnet_aux_rec_pkl_path="TFLOP-dataset/pse_results/test/end2end_results.pkl"

tokenizer_name_or_path="${exp_savepath}/${epoch_step_checkpoint}"
model_name_or_path="${tokenizer_name_or_path}"
exp_config_path="${exp_savepath}/config.yaml"
model_config_path="${exp_savepath}/${epoch_step_checkpoint}/config.json"

echo "Test..."
python test.py --tokenizer_name_or_path ${tokenizer_name_or_path} \
               --model_name_or_path ${model_name_or_path} \
               --exp_config_path ${exp_config_path} \
               --model_config_path ${model_config_path} \
               --aux_json_path $pubtabnet_aux_json_path \
               --aux_img_path $pubtabnet_aux_img_path \
               --aux_rec_pkl_path $pubtabnet_aux_rec_pkl_path \
               --batch_size 12 \
               --save_dir "${exp_savepath}/${epoch_step_checkpoint}" \
               --current_bin $bin_idx \
               --num_bins $total_bin_cnt

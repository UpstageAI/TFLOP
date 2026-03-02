EXP_NAME="pubtabnet_experiment"
EXP_VERSION="${EXP_NAME}"
RESULT_PATH="results"
TOKENIZER="hyunwoongko/asian-bart-ecjk"

uv run train.py --exp_config config/exp_configs/general_exp.yaml \
                --data_config config/exp_configs/data_pubtabnet.yaml \
                exp_version=${EXP_VERSION} \
                exp_name=${EXP_NAME} \
                result_path=${RESULT_PATH} \
                max_length=1376 \
                bbox_token_cnt=864 \
                train_batch_size=4 \
                val_batch_size=4 \
                accumulate_grad_batches=4 \
                use_OTSL=True \
                num_workers=0 \
                use_bbox_HiMulConET=True \
                lr=0.00008 \
                max_steps=250000 \
                pretrained_tokenizer_name_or_path=${TOKENIZER} \
                use_imgRoiAlign=True \
                use_RowWise_contLearning=True \
                use_ColWise_contLearning=True \
                span_coeff_mode=proportional \
                empty_cell_ptr_loss_coeff=0.5 \
                non_empty_cell_ptr_loss_coeff=0.5

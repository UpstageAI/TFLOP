import argparse
from functools import partial
import json
from multiprocessing.pool import ThreadPool
import os

from Levenshtein import distance
from omegaconf import OmegaConf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer

from tflop.datamodule.datasets.tflop import TFLOPTestDataset
from tflop.model.model.TFLOP import TFLOP
from tflop.model.model.TFLOP_Config import TFLOPConfig
from tflop.utils import custom_format_html, decode_OTSL_seq, resolve_missing_config


def evaluate_model(
    model,
    tokenizer,
    dataloader,
    config,
    model_dtype,
    current_bin=-1,
    num_bins=0,
):

    batch_lower_bound = 0
    batch_upper_bound = len(dataloader)
    if current_bin >= 0:
        assert num_bins > 0
        dataloader_binsize = len(dataloader) // num_bins
        if len(dataloader) % num_bins != 0:
            dataloader_binsize += 1
        batch_lower_bound = current_bin * dataloader_binsize
        batch_upper_bound = min((current_bin + 1) * dataloader_binsize, len(dataloader))

    result_collection = {}
    batch_index = 0
    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        if batch_index < batch_lower_bound or batch_index >= batch_upper_bound:
            batch_index += 1
            continue

        pointer_args = None
        if config.use_ptr_decoder:
            # img_tensor, input_ids, coords_int_padded, valid_coord_length, prompt_end_index, html_with_content, cell_text_collated
            image_tensors = batch[0]  # (bsz, 3, height, width)
            decoder_input_ids = batch[1]  # (bsz, text_token_length)
            coord_input_idx = batch[2]  # (bsz, bbox_token_length, 4)
            coord_input_length = batch[3]  # (bsz,)
            prompt_end_idxs = batch[4]  # (bsz,)
            html_with_content = batch[5]  # list of length==bsz
            cell_texts = batch[6]  # list of length==bsz
            file_names = batch[7]  # list of length==bsz

            pointer_args = {
                "coord_input_idx": coord_input_idx,
                "coord_input_length": coord_input_length,
            }
        else:
            raise NotImplementedError

        decoder_prompts = pad_sequence(
            [
                input_id[: end_idx + 1]
                for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )
        # Cast tensors to same dtype as model
        if model_dtype == "float16":
            image_tensors = image_tensors.half()
        elif model_dtype == "bfloat16":
            image_tensors = image_tensors.bfloat16()
        else:
            raise ValueError(f"Invalid torch dtype: {model_dtype}")

        # Move data to cuda if model is cuda
        if torch.cuda.is_available():
            image_tensors = image_tensors.cuda()
            decoder_prompts = decoder_prompts.cuda()

            if pointer_args is not None:
                pointer_args["coord_input_idx"] = pointer_args["coord_input_idx"].cuda()
                pointer_args["coord_input_length"] = pointer_args[
                    "coord_input_length"
                ].cuda()

        preds = model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
            pointer_args=pointer_args,
        )
        # preds content:
        #   - output_sequences: (bsz, M), where M is max number of tokens generated within the batch (includes BOS and <s_start> tokens)
        #   - text_to_dr_coord: (bsz, M - 2, bbox_token_cnt)

        # Get html seq with content
        pred_collection = []
        token_pred_collection = []
        raw_collection = []
        for data_i in range(preds["text_to_dr_coord"].shape[0]):
            token_id_seq = preds["output_sequences"][data_i]
            cell_text_data = cell_texts[data_i].split("<special_cell_text_sep>")
            token_seq = tokenizer.convert_ids_to_tokens(token_id_seq)
            token_pred_collection.append(token_seq)

            output_seq_tokens = decode_OTSL_seq(
                otsl_token_seq=token_seq,
                pointer_tensor=preds["text_to_dr_coord"][data_i],
                cell_text_data=cell_text_data,
            )
            pred_collection.append(output_seq_tokens)

            # Also collect raw output for sanity check
            raw_token_seq = []
            for token_pred in token_seq:
                if token_pred == "▁":
                    token_to_add = " "
                else:
                    token_to_add = token_pred.replace("▁", "")
                raw_token_seq.append(token_to_add)
            raw_token_seq = "".join(raw_token_seq)
            raw_collection.append(raw_token_seq)

            # Third, get scores
            current_batch_result = {}
            data_index = 0
            for pred, answer, token_pred in zip(
                pred_collection, html_with_content, token_pred_collection
            ):
                curr_filename = file_names[data_index]
                assert (
                    curr_filename not in current_batch_result
                ), f"Duplicate filename: {curr_filename}"

                # pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)

                pred_string, refined_pred = custom_format_html(pred, tokenizer)
                answer_string, refined_gold = custom_format_html(answer, tokenizer)

                edit_distance = distance(pred_string, answer_string) / max(
                    len(pred_string), len(answer_string)
                )
                # edit_distance = 0.0 # Removing distance calc for mem and speed measurements
                data_index += 1

                current_batch_result[curr_filename] = {
                    "pred_string": pred_string,
                    "answer_string": answer_string,
                    "edit_distance": edit_distance,
                    "token_pred": token_pred,
                }
        # Add to result collection
        for filename, result in current_batch_result.items():
            assert filename not in result_collection, f"Duplicate filename: {filename}"
            result_collection[filename] = result
        batch_index += 1
    return result_collection


def custom_load_state_dict(model, state_dict_map):
    assert len(state_dict_map) == 2
    if state_dict_map["key"] == "encoder":
        model.encoder.load_state_dict(state_dict_map["value"])
    elif state_dict_map["key"] == "decoder":
        model.decoder.load_state_dict(state_dict_map["value"])
    else:
        raise ValueError("Invalid state dict map key")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--exp_config_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--aux_json_path", type=str, default=None)
    parser.add_argument("--aux_img_path", type=str, default=None)
    parser.add_argument("--aux_rec_pkl_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="/root")
    parser.add_argument("--current_bin", type=int, default=-1)
    parser.add_argument("--num_bins", type=int, default=0)
    parser.add_argument("--use_validation", action="store_true")
    args = parser.parse_args()

    if args.current_bin >= 0:
        assert args.num_bins > 0

    # Load saved config
    exp_config = OmegaConf.load(args.exp_config_path)
    model_config = OmegaConf.load(args.model_config_path)
    print("Config file loaded")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    print("Tokenizer loaded")

    print("\nLoading model...")
    # Load pretrained model
    # 1. Initializing model
    model_config_dict = {
        k: v for k, v in exp_config.items() if k in TFLOPConfig.get_member_variables()
    }
    model_config_dict = resolve_missing_config(model_config_dict)
    data_ids = ["C-tag"]
    model = TFLOP(
        config=TFLOPConfig(**model_config_dict),
        tokenizer=tokenizer,
        data_ids=data_ids,
    )

    # 2. Load pretrained weights
    saved_state_dict = torch.load(
        os.path.join(args.model_name_or_path, "pytorch_model.bin"),
        map_location=torch.device("cpu"),
    )
    encoder_state_dict = {
        k[len("encoder.") :]: v
        for k, v in saved_state_dict.items()
        if k.startswith("encoder.")
    }
    decoder_state_dict = {
        k[len("decoder.") :]: v
        for k, v in saved_state_dict.items()
        if k.startswith("decoder.")
    }
    if len(saved_state_dict) != (len(encoder_state_dict) + len(decoder_state_dict)):
        raise ValueError("Invalid saved state dict")
    print("Loading state_dict into model...")
    with ThreadPool(2) as p:
        p.map(
            partial(custom_load_state_dict, model),
            [
                {"key": "encoder", "value": encoder_state_dict},
                {"key": "decoder", "value": decoder_state_dict},
            ],
        )
    print("Model weights loaded")

    # 3. Set model dtype, device and mode
    if model_config.torch_dtype == "float16":
        model.half()
    elif model_config.torch_dtype == "bfloat16":
        model.bfloat16()
    else:
        raise ValueError(f"Invalid torch dtype: {model_config.torch_dtype}")

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print("Model set-up complete")

    # Set-up data module
    dataset_split = "validation" if args.use_validation else "test"
    dataset = TFLOPTestDataset(
        tokenizer=tokenizer,
        split=dataset_split,
        config=exp_config,
        aux_json_path=args.aux_json_path,
        aux_img_path=args.aux_img_path,
        aux_rec_pkl_path=args.aux_rec_pkl_path,
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print("Dataset & loader setup complete. Evaluating...")
    import time

    t1 = time.time()
    result_collection = evaluate_model(
        model,
        tokenizer,
        dataloader,
        exp_config,
        model_config.torch_dtype,
        args.current_bin,
        args.num_bins,
    )
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"Evaluation complete. Time taken: {t2 - t1:.2f} seconds")

    if args.current_bin >= 0:
        save_path = os.path.join(
            args.save_dir,
            "full_model_inference_%s_%s.json" % (args.current_bin, args.num_bins),
        )
    else:
        save_path = os.path.join(args.save_dir, "full_model_inference.json")

    with open(save_path, "w", encoding="utf-8") as ff:
        json.dump(result_collection, ff, ensure_ascii=False)

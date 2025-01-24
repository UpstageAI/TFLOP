import json
import math
import os
from pathlib import Path
import random

from Levenshtein import distance
import numpy as np
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from tflop.model.model.TFLOP import TFLOP
from tflop.model.model.TFLOP_Config import TFLOPConfig
from tflop.utils import custom_format_html, resolve_missing_config


class TFLOPModelPLModule(pl.LightningModule):
    def __init__(
        self: "TFLOPModelPLModule", config, tokenizer: PreTrainedTokenizer, mode: str
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mode = mode

        model_config_dict = {
            k: v
            for k, v in self.config.items()
            if k in TFLOPConfig.get_member_variables()
        }
        model_config_dict = resolve_missing_config(model_config_dict)

        # Set-up data tag ids
        if not self.config.use_OTSL:
            raise NotImplementedError("Non-OTSL mode is deprecated")
        data_ids = ["C-tag"]

        # Set-up model
        self.model = TFLOP(
            config=TFLOPConfig(**model_config_dict),
            tokenizer=self.tokenizer,
            data_ids=data_ids,
        )
        self.load_pretrained_weights()

    def training_step(self, batch, batch_idx):
        """Training step"""

        # Sanity check -- Pointer decoder is always used in TFLOP
        assert self.config.use_ptr_decoder, "Pointer decoder is always used in TFLOP"

        # Forward pass
        model_output = self.pointer_regular_train_forward(batch)

        # Losses
        loss = model_output.loss
        token_cls_loss = model_output.token_cls_loss
        tag2coord_pointer_loss = model_output.tag2coord_pointer_loss
        tag2coord_pointer_acc = model_output.tag2coord_pointer_acc

        bbox_TableCL_loss = model_output.bbox_TableCL_loss
        rowwise_loss = model_output.rowwise_loss
        colwise_loss = model_output.colwise_loss

        self.log_dict({"train_loss": loss}, sync_dist=True)
        self.log_dict({"train_token_cls_loss": token_cls_loss}, sync_dist=True)
        self.log_dict(
            {"train_tag2coord_pointer_loss": tag2coord_pointer_loss}, sync_dist=True
        )
        self.log_dict(
            {"train_tag2coord_pointer_acc": tag2coord_pointer_acc}, sync_dist=True
        )
        self.log_dict({"train_bbox_TableCL_loss": bbox_TableCL_loss}, sync_dist=True)
        self.log_dict({"train_rowwise_loss": rowwise_loss}, sync_dist=True)
        self.log_dict({"train_colwise_loss": colwise_loss}, sync_dist=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        """Prepare for validation step"""

        super().on_validation_epoch_start()
        self.validation_step_outputs = [[]]
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step"""

        # Sanity check -- Pointer decoder is always used in TFLOP
        assert self.config.use_ptr_decoder, "Pointer decoder is always used in TFLOP"

        # Forward pass
        preds, answers, html_with_content, cell_texts = (
            self.pointer_regular_validation_forward(batch)
        )

        # Get html seq with content
        pred_collection = []
        for data_i in range(preds["output_sequences"].shape[0]):
            token_id_seq = preds["output_sequences"][data_i]
            token_seq = self.tokenizer.convert_ids_to_tokens(token_id_seq)

            if self.config.get("use_OTSL", False):
                output_seq_tokens = []
                for token_pred in token_seq:
                    if token_pred == "▁":
                        token_to_add = " "
                    else:
                        token_to_add = token_pred.replace("▁", "")
                    output_seq_tokens.append(token_to_add)
                output_seq_tokens = "".join(output_seq_tokens)
            else:
                raise NotImplementedError("Non-OTSL mode is deprecated")

            pred_collection.append(output_seq_tokens)

        # Get scores
        scores = []
        for pred, answer in zip(pred_collection, answers):
            score_set = []

            pred_string, refined_pred = custom_format_html(pred, self.tokenizer)
            answer_string, refined_gold = custom_format_html(answer, self.tokenizer)

            score_set.append(
                distance(pred_string, answer_string)
                / max(len(pred_string), len(answer_string))
            )

            ted_score_structure_only, ted_score_full = 0.0, 0.0
            score_set.append(ted_score_structure_only)
            score_set.append(ted_score_full)

            scores.append(score_set)

        self.validation_step_outputs[dataloader_idx].append(scores)

        return scores

    def on_validation_epoch_end(self):
        """Validation epoch end"""
        # Sanity check
        assert len(self.validation_step_outputs) == 1
        cnt, edit_dist_metric, ted_no_support_metric, ted_metric, val_metric = (
            [0],
            [0],
            [0],
            [0],
            [0],
        )

        for scores in self.validation_step_outputs[0]:
            cnt[0] += len(scores)

            edit_dist_metric[0] += np.sum([x[0] for x in scores])
            ted_no_support_metric[0] += np.sum([x[1] for x in scores])
            ted_metric[0] += np.sum([x[2] for x in scores])

        val_metric[0] = edit_dist_metric[0] / cnt[0]
        val_metric_name = f"val_metric_{0}th_dataset"
        self.log_dict({val_metric_name: val_metric[0]}, sync_dist=True)

        self.log_dict(
            {
                "val_metric": np.sum(edit_dist_metric) / np.sum(cnt),
                "ted_no_support": np.sum(ted_no_support_metric) / np.sum(cnt),
                "ted": np.sum(ted_metric) / np.sum(cnt),
            },
            sync_dist=True,
        )

    def configure_optimizers(self):
        """Prepare optimizer and scheduler"""

        max_iter = self.config.max_steps
        assert max_iter is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        # set warmup steps to 2% of max_iter
        num_warmup_steps = int(max_iter * 0.02)  # ~5K if max_iter=250K
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, num_warmup_steps),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        """Create cosine scheduler with warmup"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        """Save model and tokenizer"""
        save_path = (
            Path(self.config.result_path)
            / self.config.exp_name
            / self.config.exp_version
        )
        save_path = save_path / (
            "epoch_%s_step_%s" % (self.current_epoch, self.global_step)
        )
        self.model.save_pretrained(save_path)
        self.model.decoder.tokenizer.save_pretrained(save_path)

    def load_pretrained_weights(self):
        """Load pretrained weights if available"""

        if self.config.get("pretrained_model_name_or_path", False):
            loaded_state_dict = torch.load(
                os.path.join(
                    self.config.pretrained_model_name_or_path, "pytorch_model.bin"
                )
            )
            saved_config = json.load(
                open(
                    os.path.join(
                        self.config.pretrained_model_name_or_path, "config.json"
                    ),
                    "r",
                )
            )

            # First adjust saved state_dict to match that of current model, then load_state_dict

            # 1. truncate or interplolate position embeddings of donut decoder
            if self.config.max_length != saved_config["max_length"]:
                print(
                    "NOTE: max_length of pretrained model differs max_length you want to train"
                )
                weight_tensor = self.model.decoder.resize_bart_abs_pos_emb(
                    loaded_state_dict[
                        "decoder.model.model.decoder.embed_positions.weight"
                    ],
                    self.config.max_length
                    + 2,  # https://github.com/huggingface/transformers/blob/v4.11.3/src/transformers/models/mbart/modeling_mbart.py#L118-L119
                )
                weight_tensor = weight_tensor.contiguous()
                loaded_state_dict[
                    "decoder.model.model.decoder.embed_positions.weight"
                ] = weight_tensor

            # 2. adjust swin encoder if window size mismatch
            if type(self.config.input_size) == omegaconf.dictconfig.DictConfig:
                input_size_mismatch = [
                    self.config.input_size["width"],
                    self.config.input_size["height"],
                ] != saved_config["input_size"]
            else:
                input_size_mismatch = (
                    self.config.input_size != saved_config["input_size"]
                )
            window_size_mismatch = (
                self.config.window_size != saved_config["window_size"]
            )
            if input_size_mismatch or window_size_mismatch:
                print(
                    "NOTE: input_size or window_size of pretrained model differs input_size or window_size you want to train"
                )

                curr_state_dict = self.model.encoder.state_dict()
                for x in curr_state_dict:
                    if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                        pass
                    elif (
                        x.endswith("relative_position_bias_table")
                        and self.model.encoder.model.layers[0]
                        .blocks[0]
                        .attn.window_size[0]
                        != saved_config["window_size"]
                    ):
                        pos_bias = loaded_state_dict["encoder." + x].unsqueeze(0)[0]
                        old_len = int(math.sqrt(len(pos_bias)))
                        new_len = int(2 * self.config.window_size - 1)
                        pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                            0, 3, 1, 2
                        )
                        pos_bias = F.interpolate(
                            pos_bias,
                            size=(new_len, new_len),
                            mode="bicubic",
                            align_corners=False,
                        )
                        curr_state_dict[x] = (
                            pos_bias.permute(0, 2, 3, 1)
                            .reshape(1, new_len**2, -1)
                            .squeeze(0)
                        )
                    else:
                        curr_state_dict[x] = loaded_state_dict["encoder." + x]

                for swin_enc_key in curr_state_dict.keys():
                    loaded_state_dict["encoder." + swin_enc_key] = curr_state_dict[
                        swin_enc_key
                    ]

            # Now, load state dict
            encoder_state_dicts = {
                k[len("encoder.") :]: v
                for k, v in loaded_state_dict.items()
                if k.startswith("encoder.")
            }
            decoder_state_dicts = {
                k[len("decoder.") :]: v
                for k, v in loaded_state_dict.items()
                if k.startswith("decoder.")
            }

            # But first remove size_mismatched keys
            tmp_current_encoder_statedict = self.model.encoder.state_dict()
            tmp_current_decoder_statedict = self.model.decoder.state_dict()

            encoder_size_mismatched_keys = []
            encoder_keys_to_be_deleted = []
            for encoder_key in encoder_state_dicts.keys():
                if (
                    encoder_key in tmp_current_encoder_statedict
                    and tmp_current_encoder_statedict[encoder_key].shape
                    != encoder_state_dicts[encoder_key].shape
                ):
                    encoder_size_mismatched_keys.append(
                        [
                            encoder_key,
                            tmp_current_encoder_statedict[encoder_key].shape,
                            encoder_state_dicts[encoder_key].shape,
                        ]
                    )
                    encoder_keys_to_be_deleted.append(encoder_key)

            decoder_size_mismatched_keys = []
            decoder_keys_to_be_deleted = []
            for decoder_key in decoder_state_dicts.keys():
                if (
                    decoder_key in tmp_current_decoder_statedict
                    and tmp_current_decoder_statedict[decoder_key].shape
                    != decoder_state_dicts[decoder_key].shape
                ):
                    decoder_size_mismatched_keys.append(
                        [
                            decoder_key,
                            tmp_current_decoder_statedict[decoder_key].shape,
                            decoder_state_dicts[decoder_key].shape,
                        ]
                    )

                    decoder_keys_to_be_deleted.append(decoder_key)

            encoder_state_dicts = {
                k: v
                for k, v in encoder_state_dicts.items()
                if k not in encoder_keys_to_be_deleted
            }
            decoder_state_dicts = {
                k: v
                for k, v in decoder_state_dicts.items()
                if k not in decoder_keys_to_be_deleted
            }

            encoder_missing_keys, encoder_unexpected_keys = (
                self.model.encoder.load_state_dict(encoder_state_dicts, strict=False)
            )
            decoder_missing_keys, decoder_unexpected_keys = (
                self.model.decoder.load_state_dict(decoder_state_dicts, strict=False)
            )

            print("-----Size Mismatched Keys-----")
            print("Encoder:")
            if len(encoder_size_mismatched_keys) > 0:
                for key in encoder_size_mismatched_keys:
                    mismatched_keyname, curr_shape, loaded_shape = key
                    print(
                        f"{mismatched_keyname}: trying to load: {loaded_shape} -> into: {curr_shape}"
                    )
            else:
                print("None")
            print("\nDecoder:")
            if len(decoder_size_mismatched_keys) > 0:
                for key in decoder_size_mismatched_keys:
                    mismatched_keyname, curr_shape, loaded_shape = key
                    print(
                        f"{mismatched_keyname}: trying to load: {loaded_shape} -> into: {curr_shape}"
                    )
            else:
                print("None")
            print("-------------------------------")
            print("----------Missing Keys---------")
            print("Encoder:")
            if len(encoder_missing_keys) > 0:
                for key in encoder_missing_keys:
                    print(key)
            else:
                print("None")
            print("\nDecoder:")
            if len(decoder_missing_keys) > 0:
                for key in decoder_missing_keys:
                    print(key)
            else:
                print("None")
            print("-------------------------------")
            print("--------Unexpected Keys--------")
            print("Encoder:")
            if len(encoder_unexpected_keys) > 0:
                for key in encoder_unexpected_keys:
                    print(key)
            else:
                print("None")
            print("\nDecoder:")
            if len(decoder_unexpected_keys) > 0:
                for key in decoder_unexpected_keys:
                    print(key)
            else:
                print("None")
            print("-------------------------------")

    def pointer_regular_train_forward(self, batch):
        """Forward pass for regular training stage"""
        image_tensors = batch[0]  # (batch_size, 3, height, width)
        decoder_input_ids = batch[1]  # (batch_size, text_token_length)
        coord_input_idx = batch[2]  # (batch_size, bbox_token_length, 4)
        coord_input_length = batch[3]  # (batch_size,)
        decoder_token_labels = batch[4]  # (batch_size, text_token_length)
        pointer_labels = batch[
            5
        ]  # (batch_size, text_token_length - 2, bbox_token_length)
        pointer_mask_labels = batch[6]  # (batch_size, text_token_length - 2)
        bbox_coeff_tensor = batch[
            7
        ]  # (batch_size, 5, bbox_token_length, bbox_token_length)

        pointer_args = {
            "coord_input_idx": coord_input_idx,
            "coord_input_length": coord_input_length,
            "pointer_labels": pointer_labels,
            "pointer_mask_labels": pointer_mask_labels,
            "bbox_coeff_tensor": bbox_coeff_tensor,
        }

        model_output = self.model(
            image_tensors=image_tensors,
            decoder_input_ids=decoder_input_ids,
            decoder_labels=decoder_token_labels,
            pointer_args=pointer_args,
        )

        return model_output

    def pointer_regular_validation_forward(self, batch):
        """Forward pass for regular validation stage"""
        image_tensors = batch[0]  # (bsz, 3, height, width)
        decoder_input_ids = batch[1]  # (bsz, text_token_length)
        coord_input_idx = batch[2]  # (bsz, bbox_token_length, 4)
        coord_input_length = batch[3]  # (bsz,)
        prompt_end_idxs = batch[4]  # (bsz,)
        answers = batch[5]  # list of length==bsz
        pointer_labels = batch[6]  # (bsz, text_token_length - 2, bbox_token_length)
        pointer_mask_labels = batch[7]  # (bsz, text_token_length - 2)
        html_with_content = batch[8]  # list of length==bsz
        cell_texts = batch[9]  # list of length==bsz
        file_names = batch[10]  # list of length==bsz
        bbox_coeff_tensor = batch[11]  # (bsz, 5, bbox_token_length, bbox_token_length)

        pointer_args = {
            "coord_input_idx": coord_input_idx,
            "coord_input_length": coord_input_length,
            "pointer_labels": pointer_labels,
            "pointer_mask_labels": pointer_mask_labels,
        }

        decoder_prompts = pad_sequence(
            [
                input_id[: end_idx + 1]
                for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
            ],
            batch_first=True,
        )
        preds = self.model.inference(
            image_tensors=image_tensors,
            prompt_tensors=decoder_prompts,
            return_json=False,
            return_attentions=False,
            pointer_args=pointer_args,
        )

        return preds, answers, html_with_content, cell_texts


class DataPLModule(pl.LightningDataModule):
    def __init__(self: "DataPLModule", config):
        super().__init__()
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_batch_size = self.config.train_batch_size
        self.val_batch_size = self.config.val_batch_size
        self.test_batch_size = self.config.test_batch_size

        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

        for ds in [self.train_dataset, self.val_dataset, self.test_dataset]:
            if ds is not None:
                assert type(ds) == torch.utils.data.Dataset

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        return dataloader

    @staticmethod
    def seed_worker(wordker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

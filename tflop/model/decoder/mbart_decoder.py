from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from transformers import MBartConfig, MBartForCausalLM, PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from transformers.models.mbart.modeling_mbart import _expand_mask, _make_causal_mask

from tflop.loss import TableCL
from tflop.model.decoder.utils import apply_fast_mbart_decoder


class MBARTDecoder(nn.Module):
    def __init__(
        self: "MBARTDecoder",
        tokenizer: PreTrainedTokenizer,
        decoder_layer: int,
        max_length: int,
        name_or_path: str,
        max_position_embeddings: Union[int, None] = None,
        use_fast: bool = False,
        input_size: Tuple[int] = None,  # (width, height)
        bbox_token_cnt: int = None,
        max_num_row: int = None,
        max_num_col: int = None,
        use_bbox_HiMulConET: bool = False,
        use_imgRoiAlign: bool = False,
        contrastive_loss_config: dict = None,
        empty_cell_ptr_loss_coeff: float = 0.5,
        non_empty_cell_ptr_loss_coeff: float = 0.5,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_position_embeddings
            if max_position_embeddings is not None
            else max_length
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.use_fast = use_fast
        self.input_size = input_size
        self.bbox_token_cnt = bbox_token_cnt

        self.max_num_row = max_num_row
        self.max_num_col = max_num_col
        self.use_bbox_HiMulConET = use_bbox_HiMulConET
        self.use_imgRoiAlign = use_imgRoiAlign
        self.contrastive_loss_config = contrastive_loss_config
        self.empty_cell_ptr_loss_coeff = empty_cell_ptr_loss_coeff
        self.non_empty_cell_ptr_loss_coeff = non_empty_cell_ptr_loss_coeff

        self.config = MBartConfig(
            is_decoder=True,
            is_encoder_decoder=False,
            add_cross_attention=True,
            decoder_layers=self.decoder_layer,
            max_position_embeddings=self.max_position_embeddings,
            vocab_size=len(self.tokenizer),
            scale_embedding=True,
            add_final_layer_norm=True,
        )

        self.model = MBartForCausalLM(config=self.config)
        if self.use_fast:
            apply_fast_mbart_decoder(self.model)
        self.model.forward = (
            self.forward
        )  # to get cross attentions and utilize `generate` function

        self.model.config.is_encoder_decoder = True  # to get cross-attention
        self.model.model.decoder.embed_tokens.padding_idx = self.tokenizer.pad_token_id
        self.model.prepare_inputs_for_generation = self.prepare_inputs_for_inference
        self.model.model.decoder._prepare_decoder_attention_mask = (
            self._custom_prepare_decoder_attention_mask
        )
        self.get_token_ids_to_token()

        # Set up pointer decoder network parameters
        assert self.input_size is not None
        self.k_linear = nn.Linear(
            self.model.config.d_model, self.model.config.d_model, bias=False
        )
        self.q_linear = nn.Linear(
            self.model.config.d_model, self.model.config.d_model, bias=False
        )

        assert self.model.config.d_model % 4 == 0, "d_model must be divisible by 4"

        # Set up coordinate embedding, NOTE: padding_idx is set to input_size + 3, as input_size +1, +2 are used for dummy coordinates
        self.x_coord_embedding = nn.Embedding(
            self.input_size[0] + 4,
            self.model.config.d_model // 4,
            padding_idx=self.input_size[0] + 3,
        )
        self.y_coord_embedding = nn.Embedding(
            self.input_size[1] + 4,
            self.model.config.d_model // 4,
            padding_idx=self.input_size[1] + 3,
        )

        if self.use_bbox_HiMulConET:
            # Set up modules for row-wise and column-wise linear transformation
            self.rowwise_linear = nn.Linear(
                self.model.config.d_model, self.model.config.d_model, bias=False
            )
            self.colwise_linear = nn.Linear(
                self.model.config.d_model, self.model.config.d_model, bias=False
            )
            self.TableCL_loss = TableCL(temperature=0.1)

        if self.use_imgRoiAlign:
            # Set up modules for Image ROIAlignment
            self.img_downsize_scale = 32
            assert (
                self.input_size[0] == 768 and self.input_size[1] == 768
            ), "input_size must be (768, 768) when use_imgRoiAlign is True"
            self.roi_align = RoIAlign(
                output_size=(2, 2),
                spatial_scale=1 / self.img_downsize_scale,
                sampling_ratio=-1,
                aligned=False,
            )
            self.roi_proj = nn.Sequential(
                nn.Linear(self.model.config.d_model * 4, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )
            self.dummy_embed = nn.Embedding(1, self.model.config.d_model)
            self.bbox_coord_merge = nn.Sequential(
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )
            self.roi_merge = nn.Sequential(
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
                nn.ReLU(),
                nn.Linear(self.model.config.d_model, self.model.config.d_model),
            )

        if self.name_or_path is None:
            raise NotImplementedError

    def _custom_prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        """Modification of attention mask for decoder

        NOTE:
            - This function overrides the default `_prepare_decoder_attention_mask` function
            - Aims to modify the attention mask to allow bi-directional attention for prefix tokens corresponding to bbox
        """

        if self.bbox_token_cnt:
            prefix_dimension = self.bbox_token_cnt + 1  # add 1 for bos token position
        else:
            prefix_dimension = None

        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

            if (
                prefix_dimension is not None
                and combined_attention_mask.shape[-2] >= prefix_dimension
            ):
                assert (
                    combined_attention_mask.shape[-2]
                    == combined_attention_mask.shape[-1]
                ), "Only square attention masks are allowed"
                combined_attention_mask[:, :, :prefix_dimension, :prefix_dimension] = (
                    0  # Allow bi-directional for prefix_dimension tokens
                )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def prepare_inputs_for_inference(
        self: "MBARTDecoder",
        input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        past_key_values: torch.Tensor = None,
        past=None,
        use_cache: bool = None,
        attention_mask: torch.Tensor = None,
        input_coords: torch.Tensor = None,
        input_coords_length: torch.Tensor = None,
    ):
        """Custom function for preparing inputs for inference

        NOTE:
            - This function overrides the default `prepare_inputs_for_generation` function
        """

        if past is not None:
            past_key_values = past
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "encoder_hidden_states": encoder_outputs.last_hidden_state,
            "inference_mode": True,
            "input_coords": input_coords,
            "input_coords_length": input_coords_length,
        }

        return output

    def embed_coord_tensor(self, input_coord_tensor: torch.Tensor):
        """Embed coordinate tensor"""
        assert input_coord_tensor.shape[-1] == 4
        coord_embedding = torch.cat(
            [
                self.x_coord_embedding(input_coord_tensor[..., 0]),
                self.y_coord_embedding(input_coord_tensor[..., 1]),
                self.x_coord_embedding(input_coord_tensor[..., 2]),
                self.y_coord_embedding(input_coord_tensor[..., 3]),
            ],
            dim=-1,
        )

        return coord_embedding

    def get_img_roiAlign(self, encoder_hidden_states, quad_input_coords):
        """
        Get Image ROIAlign based on input coordinates

        Args:
            encoder_hidden_states: (bsz, embed_h * embed_w, d_model)
            quad_input_coords: (bsz, bbox_token_length, 4)
        """
        # convert coords to roialign
        org_dtype = encoder_hidden_states.dtype
        img_idx_tensor = (
            torch.arange(
                encoder_hidden_states.shape[0], device=encoder_hidden_states.device
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # (bsz, 1, 1)
        img_idx_tensor = img_idx_tensor.repeat(
            1, quad_input_coords.shape[1], 1
        )  # (bsz, bbox_token_length, 1)
        input_coord_with_idx = torch.cat(
            [img_idx_tensor, quad_input_coords], dim=-1
        )  # (bsz, bbox_token_length, 5)
        input_coord_with_idx = input_coord_with_idx.to(
            torch.float
        )  # (bsz, bbox_token_length, 5)
        bsz, bbox_token_cnt, _ = input_coord_with_idx.shape
        rois = input_coord_with_idx.view(-1, 5)  # (bsz * bbox_token_cnt, 5)

        # encoder_hidden_states (bsz, embed_h * embed_w, d_model) -> (bsz, d_model, embed_h, embed_w)
        embed_dim_h = int(self.input_size[1] / self.img_downsize_scale)
        embed_dim_w = int(self.input_size[0] / self.img_downsize_scale)
        feature_map = encoder_hidden_states.transpose(1, 2).view(
            bsz, encoder_hidden_states.shape[-1], embed_dim_h, embed_dim_w
        )  # (bsz, d_model, embed_h, embed_w)

        # typecast feature_map & rois to fp32
        pooled_features = self.roi_align(
            feature_map.to(torch.float), rois
        )  # (bsz * bbox_token_cnt, d_model, 2, 2)
        pooled_features = pooled_features.view(
            pooled_features.shape[0], -1
        )  # (bsz * bbox_token_cnt, d_model * 2 * 2)
        pooled_features = self.roi_proj(
            pooled_features.to(org_dtype)
        )  # (bsz * bbox_token_cnt, d_model)
        pooled_features = pooled_features.view(
            bsz, bbox_token_cnt, -1
        )  # (bsz, bbox_token_cnt, d_model)

        return pooled_features

    def forward(
        self: "MBARTDecoder",
        input_ids: torch.Tensor = None,
        input_coords: torch.Tensor = None,
        input_coords_length: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pointer_labels: Optional[torch.Tensor] = None,
        pointer_mask_labels: Optional[torch.Tensor] = None,
        bbox_coeff_tensor: Optional[torch.Tensor] = None,
        use_cache: bool = None,
        output_attentions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[torch.Tensor] = None,
        return_dict: bool = None,
        inference_mode: bool = False,
    ):
        """
        input_ids shape:            (batch_size, text_token_length)
        input_coords shape:         (batch_size, bbox_token_length, 4)
        input_coords_length shape:  (batch_size,)
        pointer_labels shape:       (batch_size, text_token_length - 2, bbox_token_length)
        pointer_mask_labels shape:  (batch_size, text_token_length - 2)
        bbox_coeff_tensor:          (batch_size, 5, bbox_token_length, bbox_token_length)
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.model.config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )

        if inference_mode:
            assert labels is None
        (
            loss,
            tag2coord_pointer_acc,
            token_cls_loss,
            tag2coord_pointer_loss,
            bbox_TableCL_loss,
        ) = (None, None, None, None, None)
        (
            rowwise_loss,
            colwise_loss,
        ) = (None, None)

        batch_size = input_ids.shape[0]
        # Separate handling for inference mode
        total_bbox_token_length = input_coords.shape[1]
        if inference_mode and past_key_values is not None:
            input_embeds = (
                self.model.model.decoder.embed_tokens(input_ids)
                * self.model.model.decoder.embed_scale
            )  # (batch_size, text_token_length, d_model)
        else:
            if self.use_imgRoiAlign:
                # Get Image ROIAlign
                img_ROIAlign = self.get_img_roiAlign(
                    encoder_hidden_states, input_coords
                )  # (batch_size, bbox_token_length, d_model)
                # A) Remove the last bbox_token_length entry from encoder_hidden_states
                img_ROIAlign = img_ROIAlign[
                    :, :-1
                ]  # (batch_size, bbox_token_length - 1, d_model)
                # B) Concat dummy bbox_token_length entry at the start
                dummy_ROIAlign = self.dummy_embed.weight.unsqueeze(0).repeat(
                    batch_size, 1, 1
                )  # (batch_size, 1, d_model)
                img_ROIAlign = torch.cat(
                    [dummy_ROIAlign, img_ROIAlign], dim=1
                )  # (batch_size, bbox_token_length, d_model)

            # Set up coordinate embedding of input coords along with addition of dummy coordinates
            batch_coord_embedding = self.embed_coord_tensor(
                input_coords[:, :-1]
            )  # (batch_size, bbox_token_length - 1, d_model)
            ## Concat dummy coordinate embedding at the start -> for cell data that has no corresponding dr_coord
            dummy_coord_embedding = torch.tensor(
                [
                    self.input_size[0] + 1,
                    self.input_size[1] + 1,
                    self.input_size[0] + 2,
                    self.input_size[1] + 2,
                ],
                dtype=input_coords.dtype,
                device=input_coords.device,
            )
            dummy_coord_embedding = self.embed_coord_tensor(
                dummy_coord_embedding.unsqueeze(0).unsqueeze(0)
            )  # (1, 1, d_model)
            dummy_coord_embedding = dummy_coord_embedding.repeat(
                batch_size, 1, 1
            )  # (batch_size, 1, d_model)
            batch_coord_embedding = torch.cat(
                [dummy_coord_embedding, batch_coord_embedding], dim=1
            )  # (batch_size, bbox_token_length, d_model)

            if self.use_imgRoiAlign:
                # Add Image ROIAlignment with coordinate embedding
                batch_coord_embedding = self.bbox_coord_merge(
                    batch_coord_embedding
                ) + self.roi_merge(
                    img_ROIAlign
                )  # (batch_size, bbox_token_length, d_model)

            # Text Embedding
            batch_text_embedding = (
                self.model.model.decoder.embed_tokens(input_ids)
                * self.model.model.decoder.embed_scale
            )  # (batch_size, text_token_length, d_model)

            # Combine all embeddings for input
            input_embeds = torch.cat(
                [
                    batch_text_embedding[:, 0:1],
                    batch_coord_embedding,
                    batch_text_embedding[:, 1:],
                ],
                dim=1,
            )  # (batch_size, max_seq, d_model)

            # Update labels
            if labels is not None:
                ignore_label = (
                    torch.zeros(
                        (batch_size, total_bbox_token_length),
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    - 100
                )
                labels = torch.cat([labels[:, 0:1], ignore_label, labels[:, 1:]], dim=1)

        if not inference_mode:
            input_embeds = input_embeds[:, :-1]  # (batch_size, max_seq-1, d_model)

        if labels is not None and not inference_mode:
            labels = labels[:, 1:]  # (batch_size, max_seq-1)

        # Derive attention mask for decoder that ignores padding bbox tokens
        if inference_mode:
            # from (bsz, 2) -> (bsz, 2 + total_bbox_token_length) 2 -> bos & s_start
            # attention_mask = torch.ones((attention_mask.shape[0], attention_mask.shape[1] + total_bbox_token_length), dtype=attention_mask.dtype, device=attention_mask.device)
            tmp_range_mask = torch.arange(
                attention_mask.shape[1] + total_bbox_token_length,
                device=attention_mask.device,
            ).unsqueeze(
                0
            )  # (1, 2 + total_bbox_token_length)
            valid_bbox_mask = tmp_range_mask <= (
                input_coords_length.unsqueeze(1) + 1
            )  # (bsz, 2 + total_bbox_token_length)
            non_bbox_valid_mask = tmp_range_mask >= (total_bbox_token_length + 1)
            attention_mask = torch.logical_or(valid_bbox_mask, non_bbox_valid_mask).to(
                attention_mask.dtype
            )  # (bsz, 2 + total_bbox_token_length)
        else:
            # NOTE: input_embeds is of shape (bsz, max_seq_length-1, d_model)
            tmp_range_mask = torch.arange(
                input_embeds.shape[1], device=input_embeds.device
            ).unsqueeze(
                0
            )  # (1, max_seq_length-1)
            valid_bbox_mask = tmp_range_mask <= (
                input_coords_length.unsqueeze(1) + 1
            )  # (bsz, max_seq_length-1)
            non_bbox_valid_mask = tmp_range_mask >= (total_bbox_token_length + 1)
            attention_mask = torch.logical_or(
                valid_bbox_mask, non_bbox_valid_mask
            ).long()  # (bsz, max_seq_length-1)

        outputs = self.model.model.decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.model.lm_head(outputs[0])  # (batch_size, max_seq-1, vocab_size)

        if labels is not None:
            # token classification loss
            token_cls_loss = self.get_token_classification_loss(logits, labels)

            tag2coord_pointer_loss = 0
            tag2coord_pointer_acc = 0
            bbox_TableCL_loss = 0
            (
                rowwise_loss,
                colwise_loss,
            ) = (0, 0)

            # Calculate pointer loss per data instance due to OOM
            sub_batchsize = 4
            num_sub_batches = batch_size // sub_batchsize
            if batch_size % sub_batchsize > 0:
                num_sub_batches += 1

            for sub_batch_i in range(num_sub_batches):
                # tag-to-coord pointer loss
                start_index = sub_batch_i * sub_batchsize
                end_index = (sub_batch_i + 1) * sub_batchsize
                curr_tag2coord_ptr_loss = self.get_tag2coord_ptr_loss(
                    output_seq=outputs[0][start_index:end_index],
                    total_bbox_token_length=total_bbox_token_length,
                    input_coords_length=input_coords_length[start_index:end_index],
                    pointer_label=pointer_labels[start_index:end_index],
                    pointer_mask_label=pointer_mask_labels[start_index:end_index],
                )

                if (
                    batch_size % sub_batchsize > 0
                    and sub_batch_i == num_sub_batches - 1
                ):
                    curr_batch_size = batch_size % sub_batchsize
                else:
                    curr_batch_size = sub_batchsize

                is_empty_loss, is_not_empty_loss, ptr_accuracy = curr_tag2coord_ptr_loss
                tag2coord_pointer_loss += (
                    (self.empty_cell_ptr_loss_coeff * is_empty_loss)
                    + (self.non_empty_cell_ptr_loss_coeff * is_not_empty_loss)
                ) * curr_batch_size
                tag2coord_pointer_acc += ptr_accuracy * curr_batch_size

                # HiMulConET loss
                if self.use_bbox_HiMulConET:
                    curr_bbox_TableCL_loss = self.get_bbox_TableCL_loss(
                        bbox_coeff_tensor=bbox_coeff_tensor[start_index:end_index],
                        output_seq=outputs[0][start_index:end_index],
                        total_bbox_token_length=total_bbox_token_length,
                        input_coords_length=input_coords_length[start_index:end_index],
                        contr_learning_config=self.contrastive_loss_config,
                    )

                    (
                        curr_rowwise_loss,
                        curr_colwise_loss,
                    ) = curr_bbox_TableCL_loss
                    curr_bbox_TableCL_loss = (
                        curr_rowwise_loss + curr_colwise_loss
                    ) * curr_batch_size
                    curr_bbox_TableCL_loss /= sum(self.contrastive_loss_config.values())
                    bbox_TableCL_loss += curr_bbox_TableCL_loss

                    rowwise_loss += curr_rowwise_loss * curr_batch_size
                    colwise_loss += curr_colwise_loss * curr_batch_size

            # Consolidate loss values
            tag2coord_pointer_acc /= batch_size
            tag2coord_pointer_loss /= batch_size
            loss = token_cls_loss + tag2coord_pointer_loss

            if self.use_bbox_HiMulConET:
                rowwise_loss /= batch_size  # For Logging purpose
                colwise_loss /= batch_size  # For Logging purpose

                bbox_TableCL_loss /= batch_size
                if loss is None:
                    loss = bbox_TableCL_loss
                else:
                    loss += bbox_TableCL_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            token_cls_loss=token_cls_loss,
            tag2coord_pointer_loss=tag2coord_pointer_loss,
            tag2coord_pointer_acc=tag2coord_pointer_acc,
            bbox_TableCL_loss=bbox_TableCL_loss,
            rowwise_loss=rowwise_loss,
            colwise_loss=colwise_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def add_special_tokens(self: "MBARTDecoder", list_of_tokens: List[str]):
        """Add special tokens to tokenizer and resize token embeddings"""
        newly_added_num = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": sorted(set(list_of_tokens))}
        )
        if newly_added_num > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        return newly_added_num

    def get_token_ids_to_token(self: "MBARTDecoder"):
        """Get token_id to token and token to token_id mapping"""
        token_id_to_token = {}
        token_to_token_id = {}
        spec_chars = [
            "<tr>",
            "</tr>",
            "<td>",
            "</td>",
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            'rowspan="',
            'colspan="',
        ]
        for c in spec_chars:
            token_id_to_token[self.tokenizer.convert_tokens_to_ids(c)] = c
            token_to_token_id[c] = self.tokenizer.convert_tokens_to_ids(c)

        self.token_id_to_token = token_id_to_token
        self.token_to_token_id = token_to_token_id

    def get_token_classification_loss(self, logits, labels, ignore_idx=-100):
        """Get token classification loss

        Args:
            logits: (batch_size, max_seq-1, vocab_size)
            labels: (batch_size, max_seq-1)
            ignore_idx: ignore index for loss calculation
        """
        loss_func = nn.CrossEntropyLoss(ignore_index=ignore_idx)
        token_cls_loss = loss_func(
            logits.reshape(-1, self.model.config.vocab_size), labels.reshape(-1)
        )

        return token_cls_loss

    def get_tag2coord_ptr_loss(
        self,
        output_seq,
        total_bbox_token_length,
        input_coords_length,
        pointer_label,
        pointer_mask_label,
    ):
        """Function to calculate tag2coord pointer loss

        Args:
            output_seq: (bsz, max_seq, d_model)
            total_bbox_token_length: total number of bbox tokens in the input sequence
            input_coords_length: number of bbox tokens in the input sequence
            pointer_label: (bsz, total_text_token_length - 2, total_bbox_token_length)
            pointer_mask_label: (bsz, total_text_token_length - 2)

        Note:
            Tag2Coord refers to the pointer network that points from text tokens to bbox tokens
        """
        # 1. calculate pointing probability
        # input_seq ->  <s><bbox1><bbox2>...<bboxN><s_start><thead><tr>....
        # output_seq -> <bbox1><bbox2>...<bboxN><s_start><thead><tr>....
        # Shape of output_seq is (bsz, max_seq, d_model)
        assert len(output_seq.shape) == 3, "output_seq must be (bsz, max_seq, d_model)"
        batch_size = output_seq.shape[0]
        key_feature = self.k_linear(
            output_seq[:, :total_bbox_token_length]
        )  # (bsz, total_bbox_token_length, d_model)
        query_feature = self.q_linear(
            output_seq[:, total_bbox_token_length + 1 :]
        )  # (bsz, total_text_token_length - 2, d_model)

        normalized_key_feature = F.normalize(
            key_feature, dim=-1
        )  # (bsz, total_bbox_token_length, d_model)
        normalized_query_feature = F.normalize(
            query_feature, dim=-1
        )  # (bsz, total_text_token_length - 2, d_model)
        data_combined_feat = torch.bmm(
            normalized_query_feature, normalized_key_feature.transpose(1, 2)
        )  # (bsz, total_text_token_length - 2, total_bbox_token_length)

        # 2. calculate loss
        if pointer_label.dtype != query_feature.dtype:
            pointer_label = pointer_label.to(query_feature.dtype)

        # First, extract out all is-data text tokens first
        # pointer_mask_label -> whether each token is data tag or not (e.g. OTSL -> C-tag)
        temperature = 0.1
        # data_combined_feat shape: (bsz, total_text_token_length - 2, total_bbox_token_length)
        # pointer_label shape: (bsz, total_text_token_length - 2, total_bbox_token_length)
        # pointer_mask_label shape: (bsz, total_text_token_length - 2)
        is_empty_loss = 0
        is_not_empty_loss = 0

        batchwise_pointing_acc = []
        for data_i in range(batch_size):
            is_data_only_pred = data_combined_feat[
                data_i, pointer_mask_label[data_i]
            ]  # (num_is_data_text_tokens, total_bbox_token_length)
            is_data_only_label = pointer_label[
                data_i, pointer_mask_label[data_i]
            ]  # (num_is_data_text_tokens, total_bbox_token_length)

            is_empty_loss += nn.BCEWithLogitsLoss()(
                is_data_only_pred[:, 0], is_data_only_label[:, 0]
            )

            is_not_empty_pred = is_data_only_pred[
                :, 1 : (input_coords_length[data_i] + 1)
            ]  # (num_is_data_text_tokens, input_coords_length)
            is_not_empty_label = is_data_only_label[
                :, 1 : (input_coords_length[data_i] + 1)
            ]  # (num_is_data_text_tokens, input_coords_length)
            valid_coords_tmp = (
                torch.sum(is_not_empty_label, 0) == 1
            )  # NOTE: While each data text token could correspond to multiple bbox tokens, each bbox token can only correspond to one data text token

            is_not_empty_pred = is_not_empty_pred / temperature
            is_not_empty_loss += nn.CrossEntropyLoss()(
                torch.transpose(is_not_empty_pred, 0, 1)[valid_coords_tmp],
                torch.argmax(
                    torch.transpose(is_not_empty_label, 0, 1)[valid_coords_tmp],
                    dim=-1,
                ),
            )

            with torch.no_grad():
                # is_not_empty_pred shape: (num_is_data_text_tokens, input_coords_length)
                pred_pointing = F.one_hot(
                    torch.argmax(is_not_empty_pred, dim=0),
                    num_classes=is_not_empty_pred.shape[0],
                ).transpose(
                    0, 1
                )  # (num_is_data_text_tokens, input_coords_length)
                pred_pointing = pred_pointing[:, valid_coords_tmp]

                gold_pointing = (
                    is_not_empty_label  # (num_is_data_text_tokens, input_coords_length)
                )
                gold_pointing = gold_pointing[:, valid_coords_tmp]

                equiv_tns = (
                    pred_pointing == gold_pointing
                )  # (num_is_data_text_tokens, input_coords_length)

                token_wise_equivalence = torch.sum(equiv_tns, dim=-1) == torch.sum(
                    valid_coords_tmp
                )  # (num_is_data_text_tokens)
                batchwise_pointing_acc.append(
                    torch.sum(token_wise_equivalence).float()
                    / token_wise_equivalence.shape[0]
                )

        is_not_empty_loss = is_not_empty_loss / batch_size
        is_empty_loss = is_empty_loss / batch_size
        pointing_acc = torch.mean(torch.stack(batchwise_pointing_acc, dim=0))

        return is_empty_loss, is_not_empty_loss, pointing_acc

    def get_bbox_TableCL_loss(
        self,
        bbox_coeff_tensor,
        output_seq,
        total_bbox_token_length,
        input_coords_length,
        contr_learning_config,
    ):
        """Function to calculate Contrastive Learning loss for bbox tokens

        Args:
            bbox_coeff_tensor: (batch_size, 5, bbox_token_length, bbox_token_length)
            output_seq: (batch_size, max_seq, d_model)
            total_bbox_token_length: total number of bbox tokens in the input sequence
            input_coords_length: number of bbox tokens in the input sequence
            contr_learning_config: configuration for contrastive learning
        """
        (
            rowwise_loss,
            colwise_loss,
        ) = (0, 0)
        bbox_feature_output = output_seq[
            :, :total_bbox_token_length
        ]  # (batch_size, total_bbox_token_length, d_model)

        if contr_learning_config["use_RowWise_contLearning"]:
            rowwise_feature = self.rowwise_linear(bbox_feature_output)
            rowwise_feature = F.normalize(rowwise_feature, dim=-1)
            coeff_index = (
                sum(
                    [
                        contr_learning_config["use_RowWise_contLearning"],
                    ]
                )
                - 1
            )
            rowwise_mask = bbox_coeff_tensor[:, coeff_index : (coeff_index + 1)]
            rowwise_loss = self.TableCL_loss(
                features=rowwise_feature,
                masks=rowwise_mask,
                input_coords_length=input_coords_length,
            )

        if contr_learning_config["use_ColWise_contLearning"]:
            colwise_feature = self.colwise_linear(bbox_feature_output)
            colwise_feature = F.normalize(colwise_feature, dim=-1)
            coeff_index = (
                sum(
                    [
                        contr_learning_config["use_RowWise_contLearning"],
                        contr_learning_config["use_ColWise_contLearning"],
                    ]
                )
                - 1
            )
            colwise_mask = bbox_coeff_tensor[:, coeff_index : (coeff_index + 1)]
            colwise_loss = self.TableCL_loss(
                features=colwise_feature,
                masks=colwise_mask,
                input_coords_length=input_coords_length,
            )

        return (
            rowwise_loss,
            colwise_loss,
        )

    @staticmethod
    def resize_bart_abs_pos_emb(weight: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        Resize position embeddings
        Truncate if sequence length of Bart backbone is greater than given max_length,
        else interpolate to max_length
        """
        if weight.shape[0] > max_length:
            weight = weight[:max_length, ...]
        else:
            weight = (
                F.interpolate(
                    weight.permute(1, 0).unsqueeze(0),
                    size=max_length,
                    mode="linear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 0)
            )
        return weight

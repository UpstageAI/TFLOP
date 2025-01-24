import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.file_utils import ModelOutput

from tflop.model.decoder.mbart_decoder import MBARTDecoder
from tflop.model.model.TFLOP_Config import TFLOPConfig
from tflop.model.visual_encoder.swin import SwinEncoder


class TFLOP(PreTrainedModel):
    config_class = TFLOPConfig
    base_model_prefix = "tflop"

    def __init__(
        self: "TFLOP",
        config: TFLOPConfig,
        tokenizer: PreTrainedTokenizer,
        data_ids: list = None,
    ):
        super().__init__(config)
        self.config = config
        self.tokenizer = tokenizer

        # Setup Encoder
        swin_input_size = (
            self.config.input_size[1],
            self.config.input_size[0],
        )  # NOTE: Swin input is (height, width)
        self.encoder = SwinEncoder(
            input_size=swin_input_size,
            align_long_axis=self.config.align_along_axis,
            window_size=self.config.window_size,
            name_or_path=self.config.name_or_path,
            encoder_layer=self.config.encoder_layer,
        )

        # Setup Decoder
        contrastive_loss_config = {
            "use_RowWise_contLearning": self.config.use_RowWise_contLearning,
            "use_ColWise_contLearning": self.config.use_ColWise_contLearning,
        }
        self.decoder = MBARTDecoder(
            tokenizer=self.tokenizer,
            decoder_layer=self.config.decoder_layer,
            max_length=self.config.max_length,
            name_or_path=self.config.name_or_path,
            max_position_embeddings=self.config.max_position_embeddings,
            use_fast=self.config.use_fast_decoder,
            input_size=self.config.input_size,
            bbox_token_cnt=self.config.bbox_token_cnt,
            max_num_row=self.config.max_num_row,
            max_num_col=self.config.max_num_col,
            use_bbox_HiMulConET=self.config.use_bbox_HiMulConET,
            use_imgRoiAlign=self.config.use_imgRoiAlign,
            contrastive_loss_config=contrastive_loss_config,
            empty_cell_ptr_loss_coeff=self.config.empty_cell_ptr_loss_coeff,
            non_empty_cell_ptr_loss_coeff=self.config.non_empty_cell_ptr_loss_coeff,
        )

        self.data_ids = [
            self.tokenizer.convert_tokens_to_ids(token) for token in data_ids
        ]

    def forward(
        self: "TFLOP",
        image_tensors: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_labels: torch.Tensor,
        pointer_args: dict = None,
    ) -> ModelOutput:
        # vision encoding
        encoder_outputs = self.encoder(
            image_tensors
        )  # image_tensors: (bsz, 3, 768, 768), encoder_outptus: (bsz, 24*24, 1024)

        # text decoding
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            input_coords=pointer_args["coord_input_idx"],
            input_coords_length=pointer_args["coord_input_length"],
            encoder_hidden_states=encoder_outputs,
            labels=decoder_labels,
            pointer_labels=pointer_args["pointer_labels"],
            pointer_mask_labels=pointer_args["pointer_mask_labels"],
            bbox_coeff_tensor=pointer_args["bbox_coeff_tensor"],
        )

        return decoder_outputs

    def inference(
        self: "TFLOP",
        image_tensors: torch.Tensor,
        prompt_tensors: torch.Tensor,
        return_json: bool = True,
        return_attentions: bool = False,
        pointer_args: dict = None,
        return_last_hidden_state: bool = False,
    ):
        """
        Perform inference using the TFLOP model.

        This method processes input image and prompt tensors through the TFLOP model's
        vision encoder and text decoder to produce the desired output.

        Args:
            image_tensors (torch.Tensor): Tensor representing the input images to the vision encoder.
            prompt_tensors (torch.Tensor): Tensor representing the prompt sequences for the text decoder.
            return_attentions (bool, optional): If True, the method returns attention maps from the decoder. Default is False.
            pointer_args (dict, optional):
                A dictionary containing arguments required for pointer-based decoding.
                Must include "coord_input_idx" and "coord_input_length".
            return_last_hidden_state (bool, optional):
                If True, the last hidden state of the decoder will be included in the output. Default is False.

        Returns:
            dict: A dictionary containing the following keys:
                - "output_sequences": Decoded output sequences (torch.Tensor).
                - "text_to_dr_coord": Text-to-detection-region coordinate predictions (torch.Tensor).
                - "last_hidden_state" (optional): The last hidden state of the decoder, if requested.
                - "attention" (optional): A dictionary with attention maps, if requested, containing:
                    - "self_attentions": Self-attention maps from the decoder.
                    - "cross_attentions": Cross-attention maps between encoder and decoder.

        Raises:
            ValueError: If either `image_tensors` or `prompt_tensors` is None.
            ValueError: If `pointer_args` is not provided or missing required keys.

        """

        if image_tensors is None:
            raise ValueError("image_tensors must be provided.")
        if self.device.type == "cuda":
            # image_tensors = image_tensors.half()
            image_tensors = image_tensors.to(self.device)

        if prompt_tensors is None:
            raise ValueError("prompt_tensors must be provided.")
        prompt_tensors = prompt_tensors.to(self.device)

        # vision encoding
        last_hidden_state = self.encoder(image_tensors)
        if self.device.type != "cuda":
            last_hidden_state = last_hidden_state.to(torch.float32)
        encoder_outputs = ModelOutput(
            last_hidden_state=last_hidden_state, attentions=None
        )

        # Set up vision encoder & prompt tensor for decoding
        if len(encoder_outputs.last_hidden_state.size()) == 1:
            encoder_outputs.last_hidden_state = (
                encoder_outputs.last_hidden_state.unsqueeze(0)
            )
        if len(prompt_tensors.size()) == 1:
            prompt_tensors = prompt_tensors.unsqueeze(0)

        # text decoding
        if pointer_args is not None:
            decoder_output = self.decoder.model.generate(
                decoder_input_ids=prompt_tensors,
                encoder_outputs=encoder_outputs,
                max_length=(self.config.max_length - self.config.bbox_token_cnt),
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_attentions=return_attentions,
                output_scores=True,
                output_hidden_states=True,
                input_coords=pointer_args["coord_input_idx"],
                input_coords_length=pointer_args["coord_input_length"],
            )
            last_hidden_state_collection = [
                tok_pos_output[-1]
                for tok_pos_output in decoder_output.decoder_hidden_states
            ]
            last_hidden_state_collection = torch.cat(
                last_hidden_state_collection, dim=1
            )  # (bsz, seq_len-1, d_model)

            text_to_dr_point_pred = self.get_dr_point_pred(
                last_hidden_state_collection, decoder_output.sequences
            )  # (bsz, text_token_cnt-2, bbox_token_cnt)
            output = {
                "output_sequences": decoder_output.sequences,
                "text_to_dr_coord": text_to_dr_point_pred,
            }

            if return_last_hidden_state:
                output["last_hidden_state"] = last_hidden_state_collection

        else:
            raise ValueError("pointer_args must be provided.")

        if return_attentions:
            output["attention"] = {
                "self_attentions": decoder_output.decoder_attentions,
                "cross_attentions": decoder_output.cross_attentions,
            }

        return output

    def get_dr_point_pred(self, last_hidden_state_collection, decoder_output_sequences):
        """
        get pointer tensor from hidden state and decoder output
        Args:
            last_hidden_state_collection(batch_size, seq_len-1, d_model):
            decoder_output_sequences(batch_size, seq_len-1):
        Returns:
            combined_feature(bsz, text_token_cnt-2, bbox_token_cnt):
        """

        query_feature = self.decoder.q_linear(
            last_hidden_state_collection[:, (1 + self.config.bbox_token_cnt) :]
        )  # (bsz, text_token_cnt - 2, d_model)
        key_feature = self.decoder.k_linear(
            last_hidden_state_collection[:, : self.config.bbox_token_cnt]
        )  # (bsz, bbox_token_cnt, d_model)

        combined_feature = []

        batch_size = query_feature.shape[0]
        iter_size = 8
        for i in range(0, batch_size, iter_size):
            norm_query_feature = torch.nn.functional.normalize(
                query_feature[i : i + iter_size], dim=-1
            )  # (*bsz, text_token_cnt-2, d_model)
            norm_key_feature = torch.nn.functional.normalize(
                key_feature[i : i + iter_size], dim=-1
            )  # (*bsz, bbox_token_cnt, d_model)
            tmp_feature = torch.bmm(
                norm_query_feature, norm_key_feature.transpose(1, 2)
            )  # (*bsz, text_token_cnt-2, bbox_token_cnt)
            combined_feature.append(tmp_feature)

        combined_feature = torch.cat(
            combined_feature, dim=0
        )  # (bsz, text_token_cnt-2, bbox_token_cnt)

        coord_logits = combined_feature[
            :, :, 1:
        ]  # (bsz, text_token_cnt-2, bbox_token_cnt-1)
        text_token_seq = decoder_output_sequences[:, 2:]  # (bsz, text_token_cnt-2)
        is_data_tensor = torch.zeros(
            (text_token_seq.shape[0], text_token_seq.shape[1]),
            dtype=torch.bool,
            device=text_token_seq.device,
        )  # (bsz, text_token_cnt-2)
        for data_id in self.data_ids:
            is_data_tensor = torch.logical_or(
                is_data_tensor, text_token_seq == data_id
            )  # (bsz, text_token_cnt-2)

        coord_logits[~is_data_tensor] = float(
            "-inf"
        )  # (bsz, text_token_cnt-2, bbox_token_cnt-1)
        coord_one_hot = torch.nn.functional.one_hot(
            torch.argmax(coord_logits, dim=1), num_classes=coord_logits.shape[1]
        ).transpose(
            1, 2
        )  # (bsz, text_token_cnt-2, bbox_token_cnt-1)

        # Find data where it is empty
        is_empty = torch.sum(coord_one_hot, dim=-1) == 0  # (bsz, text_token_cnt-2)
        is_empty = is_empty.unsqueeze(-1).to(
            coord_one_hot.dtype
        )  # (bsz, text_token_cnt-2, 1)

        combined_feature = torch.cat(
            [is_empty, coord_one_hot], dim=-1
        )  # (bsz, text_token_cnt-2, bbox_token_cnt)

        return combined_feature

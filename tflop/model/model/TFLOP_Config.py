import os
from typing import Tuple, Union

import omegaconf
from transformers import PretrainedConfig


class TFLOPConfig(PretrainedConfig):
    model_type = "tflop"

    def __init__(
        self: "TFLOPConfig",
        input_size: dict = {"height": 1280, "width": 960},
        align_along_axis: bool = False,
        window_size: int = 10,
        encoder_layer: Tuple[int] = (2, 2, 14, 2),
        decoder_layer: int = 4,
        max_position_embeddings: int = None,
        max_length: int = 768,
        name_or_path: Union[str, bytes, os.PathLike] = "",
        use_fast_decoder: bool = False,
        use_ptr_decoder: bool = False,
        bbox_token_cnt: int = None,
        use_cell_bbox: bool = False,
        max_num_row: int = 40,
        max_num_col: int = 40,
        use_bbox_HiMulConET: bool = False,
        use_imgRoiAlign: bool = False,
        use_RowWise_contLearning: bool = False,
        use_ColWise_contLearning: bool = False,
        empty_cell_ptr_loss_coeff: float = 0.5,
        non_empty_cell_ptr_loss_coeff: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        if type(input_size) in [dict, omegaconf.dictconfig.DictConfig]:
            self.input_size = (
                input_size["width"],
                input_size["height"],
            )  # Set to default (width, height)
        else:
            self.input_size = input_size
        self.align_along_axis = align_along_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.max_position_embeddings = (
            max_length if max_position_embeddings is None else max_position_embeddings
        )
        self.max_length = max_length
        self.name_or_path = name_or_path
        self.use_fast_decoder = use_fast_decoder
        self.use_ptr_decoder = use_ptr_decoder
        self.bbox_token_cnt = bbox_token_cnt
        self.use_cell_bbox = use_cell_bbox
        self.max_num_row = max_num_row
        self.max_num_col = max_num_col
        self.use_bbox_HiMulConET = use_bbox_HiMulConET
        self.use_imgRoiAlign = use_imgRoiAlign
        self.use_RowWise_contLearning = use_RowWise_contLearning
        self.use_ColWise_contLearning = use_ColWise_contLearning
        self.empty_cell_ptr_loss_coeff = empty_cell_ptr_loss_coeff
        self.non_empty_cell_ptr_loss_coeff = non_empty_cell_ptr_loss_coeff

    @classmethod
    def get_member_variables(cls):
        return [
            "input_size",
            "align_along_axis",
            "window_size",
            "encoder_layer",
            "decoder_layer",
            "max_position_embeddings",
            "max_length",
            "name_or_path",
            "use_fast_decoder",
            "use_ptr_decoder",
            "bbox_token_cnt",
            "use_cell_bbox",
            "max_num_row",
            "max_num_col",
            "use_bbox_HiMulConET",
            "use_imgRoiAlign",
            "use_RowWise_contLearning",
            "use_ColWise_contLearning",
            "empty_cell_ptr_loss_coeff",
            "non_empty_cell_ptr_loss_coeff",
        ]

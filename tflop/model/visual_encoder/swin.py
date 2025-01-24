import math
import os
from typing import Tuple, Union

import timm
from timm.models.swin_transformer import SwinTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinEncoder(nn.Module):
    r"""
    Donut encoder based on SwinTransformer
    Set the initial weights and configuration with a pretrained SwinTransformer and then
    modify the detailed configurations as a Donut Encoder

    Args:
        input_size: Input image size (height, width)
        align_long_axis: Whether to rotate image if height is greater than width
        window_size: Window size(=patch size) of SwinTransformer
        encoder_layer: Number of layers of SwinTransformer encoder
        name_or_path: Name of a pretrained model name either registered in huggingface.co. or saved in local.
                      otherwise, `swin_base_patch4_window12_384` will be set (using `timm`).
    """

    def __init__(
        self: "SwinEncoder",
        input_size: Tuple[int],
        align_long_axis: bool,
        window_size: int,
        name_or_path: Union[str, bytes, os.PathLike] = None,
        encoder_layer: Tuple[int] = (2, 2, 14, 2),
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        self.name_or_path = name_or_path

        self.model = SwinTransformer(
            img_size=self.input_size,
            depths=self.encoder_layer,
            window_size=self.window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )

        self.model.norm = None

        # weight init with swin
        if self.name_or_path is None:
            raise NotImplementedError
            swin_state_dict = timm.create_model(
                "swin_base_patch4_window12_384", pretrained=True
            ).state_dict()
            state_dict = self.model.state_dict()
            for x in state_dict:
                if x.endswith("relative_position_index") or x.endswith("attn_mask"):
                    pass
                elif (
                    x.endswith("relative_position_bias_table")
                    and self.model.layers[0].blocks[0].attn.window_size[0] != 12
                ):
                    pos_bias = swin_state_dict[x].unsqueeze(0)[0]
                    old_len = int(math.sqrt(len(pos_bias)))
                    new_len = int(2 * self.window_size - 1)
                    pos_bias = pos_bias.reshape(1, old_len, old_len, -1).permute(
                        0, 3, 1, 2
                    )
                    pos_bias = F.interpolate(
                        pos_bias,
                        size=(new_len, new_len),
                        mode="bicubic",
                        align_corners=False,
                    )
                    state_dict[x] = (
                        pos_bias.permute(0, 2, 3, 1)
                        .reshape(1, new_len**2, -1)
                        .squeeze(0)
                    )
                else:
                    state_dict[x] = swin_state_dict[x]
            self.model.load_state_dict(state_dict)

    def forward(self: "SwinEncoder", image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_channels, height, width)
        """
        image_tensors = self.model.patch_embed(image_tensors)
        # image_tensors = self.model.pos_drop(image_tensors) # can be removed as long as drop_rate is not initialized or set to 0.0 in SwinTransformer

        return self.model.layers(image_tensors)

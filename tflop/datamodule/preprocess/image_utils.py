from typing import List, Tuple

import PIL
from PIL import ImageOps
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
from torchvision import transforms
from torchvision.transforms.functional import resize


def convert_PIL_to_tensor(
    input_image: PIL.Image.Image, normalize_channels: bool = True
):
    """
    Converts a PIL Image to a PyTorch tensor and optionally normalizes the channels.

    Args:
        input_image (PIL.Image.Image): The input image in PIL format.
        normalize_channels (bool, optional): If True, normalizes the channels using the ImageNet
                                             default mean and standard deviation. Defaults to True.

    Returns:
        torch.Tensor: The converted image as a PyTorch tensor.
    """
    tensor_img = transforms.ToTensor()(input_image)
    if normalize_channels:
        mean_tensor = torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
        std_tensor = torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1)
        tensor_img = (tensor_img - mean_tensor) / std_tensor

    return tensor_img


def prepare_image_tensor(
    input_image: PIL.Image.Image,
    target_img_size: tuple,
    random_padding: bool = False,
    normalize_channels: bool = True,
):
    """
    Prepares an image tensor by resizing, padding, and converting a PIL Image to a PyTorch tensor.

    Args:
        input_image (PIL.Image.Image): The input image in PIL format.
        target_img_size (tuple): The target size of the image as a tuple (width, height).
        random_padding (bool, optional): If True, applies random padding to the image. If False,
                                         centers the image with padding. Defaults to False.
        normalize_channels (bool, optional): If True, normalizes the channels using the ImageNet
                                             default mean and standard deviation. Defaults to True.

    Returns:
        torch.Tensor: The prepared image as a PyTorch tensor.
        tuple: The original size of the input image as a tuple (width, height).
        tuple: The padding dimensions applied to the image as a tuple (left, top, right, bottom).
    """
    original_size = input_image.size

    # Resize image
    target_width, target_height = target_img_size
    try:
        resized_img = resize(
            input_image.convert("RGB"), min(target_img_size)
        )  # Resized with smaller edge = min(target_img_size)
    except:
        print("Error resizing image: ", input_image.filename)
        raise ValueError("Error in resizing image.")
    resized_img.thumbnail(
        size=(target_width, target_height)
    )  # NOTE: thumbnail size is (width, height)

    # Pad image
    delta_width = target_width - resized_img.size[0]
    delta_height = target_height - resized_img.size[1]
    if random_padding:
        pad_width = np.random.randint(low=0, high=(delta_width + 1))
        pad_height = np.random.randint(low=0, high=(delta_height + 1))
    else:
        # Center image if not random padding
        pad_width = delta_width // 2
        pad_height = delta_height // 2
    padding_dims = (
        pad_width,
        pad_height,
        delta_width - pad_width,
        delta_height - pad_height,
    )
    padded_img = ImageOps.expand(resized_img, padding_dims)

    # Convert to tensor
    tensor_img = convert_PIL_to_tensor(
        padded_img, normalize_channels=normalize_channels
    )

    return tensor_img, original_size, padding_dims

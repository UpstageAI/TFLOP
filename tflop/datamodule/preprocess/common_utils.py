import random
from typing import List, Tuple

import torch

from tflop.datamodule.preprocess.hi_mul_con_table import (
    get_columnwise_HiMulConET_Coeff,
    get_rowwise_HiMulConET_Coeff,
)


def int_convert_and_pad_coords(
    coords: List[Tuple], padding_coord: int = -1, max_length: int = None
):
    """Returns int() coords padded with padding_coord to max_length.

    Args:
        coords (List[Tuple]): List of coords, where each coord is a tuple of 4 float values.
        padding_coord (int, optional): Padding value. Defaults to -1.
        max_length (int, optional): Maximum length of the output. Defaults to None.

    Returns:
        tensor of shape: (bbox_token_cnt, 4)
            e.g. (768, 4)
    """
    int_coords = [[int(x) for x in coord] for coord in coords]
    if max_length is None:
        return torch.tensor(int_coords)
    else:
        if type(padding_coord) == list:
            padd_int_coord = padding_coord
        else:
            padd_int_coord = [
                padding_coord,
                padding_coord,
                padding_coord,
                padding_coord,
            ]

        if len(int_coords) > max_length:
            return torch.tensor(int_coords[:max_length])
        else:
            for _ in range(max_length - len(int_coords)):
                int_coords.append(padd_int_coord)
            return torch.tensor(int_coords)


def convert_gold_coords(gold_coords: List[str]):
    """Converts gold_coords into dict of coords, isFilled, text.

    NOTE:
        - isFilled is True only if the cell is filled with data & bbox is valid
        - isFilled is changed to False if bbox is not valid (i.e. width or height is 0)

    Returns:
        - gold_coord_dict: dict of coords, isFilled, text
    """
    if gold_coords is None:
        return None

    gold_coord_dict = {"coords": [], "isFilled": [], "text": []}

    def is_bbox_invalid(bbox_coord_value):
        """Check if the bounding box is invalid (i.e., width or height is near 0)."""
        x1, y1, x2, y2 = bbox_coord_value
        if abs(x1 - x2) < 0.00001 or abs(y1 - y2) < 0.00001:
            return True
        return False

    for coord in gold_coords:
        split_content = coord.split(" ")
        four_coord_values = [float(x) for x in split_content[:4]]
        gold_coord_dict["coords"].append(four_coord_values)

        if is_bbox_invalid(four_coord_values):
            gold_coord_dict["isFilled"].append(False)
        else:
            gold_coord_dict["isFilled"].append(int(split_content[4]) == 2)

        gold_coord_dict["text"].append(" ".join(split_content[5:]))

    return gold_coord_dict


def generate_filled_html(gold_text_list, is_cell_filled, org_html_list):
    """Insert cell filled text within html tags to form filled html sequence.

    Args:
        gold_text_list (List[str]): List of gold text for each cell
        is_cell_filled (List[bool]): List of bool indicating if the cell is filled with data
        org_html_list (List[str]): List of html tags
    """
    filled_html_list = []
    data_index = 0

    for org_html_tag in org_html_list:
        if org_html_tag != "</td>":
            filled_html_list.append(org_html_tag)
        else:
            if len(gold_text_list[data_index]) > 0 and is_cell_filled[data_index]:
                filled_html_list.append(gold_text_list[data_index])
            filled_html_list.append(org_html_tag)
            data_index += 1

    filled_html = "".join(filled_html_list)
    return filled_html


def rescale_bbox(
    list_of_coords: List[Tuple],
    org_img_size: Tuple,
    new_img_size: Tuple,
    padding_dims: Tuple,
):
    """Rescale bounding box coordinates to new image size with padding dimensions.

    Args:
        list_of_coords (List[Tuple]): A list of bounding box coordinates as tuples (x1, y1, x2, y2).
        org_img_size (Tuple): The original size of the image as a tuple (width, height).
        new_img_size (Tuple): The new size of the image as a tuple (width, height).
        padding_dims (Tuple): The padding dimensions applied to the image as a tuple (left, top, right, bottom).

    NOTE:
        This function assumes quad-coord format (x1, y1, x2, y2)
    """
    rescaled_coords = []
    for coord in list_of_coords:
        x1, y1, x2, y2 = coord

        # Get width & height scales
        img_only_width = new_img_size[0] - padding_dims[0] - padding_dims[2]
        img_only_height = new_img_size[1] - padding_dims[1] - padding_dims[3]
        width_scale = img_only_width / org_img_size[0]
        height_scale = img_only_height / org_img_size[1]

        # Scale coords
        x1 = (x1 * width_scale) + padding_dims[0]
        x2 = (x2 * width_scale) + padding_dims[0]
        y1 = (y1 * height_scale) + padding_dims[1]
        y2 = (y2 * height_scale) + padding_dims[1]

        # Clipping coordinates to image size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(new_img_size[0], x2)
        y2 = min(new_img_size[1], y2)

        rescaled_coords.append([x1, y1, x2, y2])

    return rescaled_coords


def get_dr_pointer_label(
    dr_coords: dict,
    gold_coord_isFilled: List[bool],
    cell_shuffle_rate: float,
    input_ids: torch.IntTensor,
    data_ids: List[int],
    bbox_token_cnt: int,
    org_img_size: Tuple,
    new_img_size: Tuple,
    padding_dims: Tuple,
    coeff_tensor_args: dict,
):
    """
    This function generates the pointer label when using detection coord information (i.e. dr_coords).

    Args:
        dr_coords dict: dict containing detection coord information
        gold_coord_isFilled list(bool): list of bool indicating if <td> is filled with data
        cell_shuffle_rate float: rate of shuffling cells (for training mode only)
        input_ids torch.IntTensor: input_ids of shape (seq_length)
        data_ids list(int): list of token_ids which correspond to data tokens (e.g. <td> for html)
        bbox_token_cnt int: number of tokens in sequence dedicated for bbox coord representation (e.g. 768)
        org_img_size Tuple: original image size as a tuple (width, height)
        new_img_size Tuple: new image size as a tuple (width, height)
        padding_dims Tuple: padding dimensions as a tuple (left, top, right, bottom)
        coeff_tensor_args dict: arguments for coefficient tensor generation
    """
    # shared variable set-up
    seq_length = input_ids.shape[0]

    # 1. First get bool tensor indicating which of the input_ids correspond to data_ids
    is_data_tensor = torch.zeros(
        (seq_length - bbox_token_cnt - 1), dtype=torch.bool
    )  # -1 for bos token (causal supervision)
    for data_id in data_ids:
        is_data_tensor = torch.logical_or(
            is_data_tensor, input_ids[1:-bbox_token_cnt] == data_id
        )

    # 2. Processing detection coord information
    # 2.1 Set-up filled-text only coords, text and indices
    filtered_dr_coords = []
    filtered_texts = []
    filtered_org_index_list = []

    dr_coord_keys = [int(x) for x in list(dr_coords.keys())]
    for tmp_idx in sorted(dr_coord_keys):
        rescaled_coords = rescale_bbox(
            list_of_coords=dr_coords[str(tmp_idx)][0],
            org_img_size=org_img_size,
            new_img_size=new_img_size,
            padding_dims=padding_dims,
        )
        filtered_dr_coords.extend(rescaled_coords)
        filtered_texts.append(dr_coords[str(tmp_idx)][2])
        filtered_texts += [""] * (len(rescaled_coords) - 1)
        filtered_org_index_list.extend(
            [int(dr_coords[str(tmp_idx)][1])] * len(rescaled_coords)
        )

    # 2.2 Serialize filtered_dr_coords from top-left to bottom-right
    serialized_dr_cell_coords, serialized_dr_cell_texts, serialized_org_index_list = (
        serialize_bbox_top_left_bottom_right(filtered_dr_coords, filtered_texts)
    )
    filtered_org_index_list = [
        filtered_org_index_list[i] for i in serialized_org_index_list
    ]

    # 2.3 In event that no. of coord entries exceed bbox_token_cnt, slice it. NOTE -1 as first bbox is used for empty cell pointing
    if len(serialized_dr_cell_coords) > (bbox_token_cnt - 1):
        filtered_dr_coords = serialized_dr_cell_coords[: bbox_token_cnt - 1]
        filtered_texts = serialized_dr_cell_texts[: bbox_token_cnt - 1]
        filtered_org_index_list = filtered_org_index_list[: bbox_token_cnt - 1]
    else:
        filtered_dr_coords = serialized_dr_cell_coords
        filtered_texts = serialized_dr_cell_texts

    # 2.4 Shuffle filtered_dr_coords at cell_shuffle_rate
    if random.random() < cell_shuffle_rate:
        zipped_tmp = list(
            zip(filtered_org_index_list, filtered_dr_coords, filtered_texts)
        )
        random.shuffle(zipped_tmp)
        new_org_index_list, new_filtered_dr_coords, new_filtered_texts = zip(
            *zipped_tmp
        )
    else:
        new_org_index_list = filtered_org_index_list
        new_filtered_dr_coords = filtered_dr_coords
        new_filtered_texts = filtered_texts

    # 3. Get pointer label
    pointer_label = []
    dataIndex2bboxIndex = {}
    data_cell_index = 0
    for i in range(is_data_tensor.shape[0]):
        tmp_label = torch.zeros((bbox_token_cnt), dtype=input_ids.dtype)

        if is_data_tensor[i]:  # this is a data token, i.e. either '<td>' or '<td '
            if gold_coord_isFilled[
                data_cell_index
            ]:  # this cell is supposed to be filled with content
                if (
                    data_cell_index in new_org_index_list
                ):  # NOTE: new_org_index_list contains indicies of data-cells
                    # multiple index match could occur for dr coords
                    tmp_indices = [
                        i
                        for i, x in enumerate(new_org_index_list)
                        if x == data_cell_index
                    ]
                    for tmp_indx in tmp_indices:
                        tmp_label[tmp_indx + 1] = 1
                    dataIndex2bboxIndex[i] = tmp_indices
                else:
                    # corresponding bbox missing due to exceeding bbox_token_cnt
                    is_data_tensor[i] = False
            else:
                # This is a table data with no content
                tmp_label[0] = 1

            data_cell_index += 1

        pointer_label.append(tmp_label)

    pointer_label = torch.stack(
        pointer_label, dim=0
    )  # shape: (seq_length - bbox_token_cnt - 1, bbox_token_cnt)

    # Coefficient tensor for HiMulConET
    rowwise_coeff_tensor = get_rowwise_HiMulConET_Coeff(
        sliced_input_ids=input_ids[1:-bbox_token_cnt],
        tokenizer=coeff_tensor_args["tokenizer"],
        bbox_token_cnt=bbox_token_cnt,
        is_data_tensor=is_data_tensor,
        tag2coord_map=dataIndex2bboxIndex,
        rep_mode=coeff_tensor_args["rep_mode"],
        rowspan_coeff_mode=coeff_tensor_args["rowspan_coeff_mode"],
    )
    colwise_coeff_tensor = get_columnwise_HiMulConET_Coeff(
        sliced_input_ids=input_ids[1:-bbox_token_cnt],
        tokenizer=coeff_tensor_args["tokenizer"],
        bbox_token_cnt=bbox_token_cnt,
        is_data_tensor=is_data_tensor,
        tag2coord_map=dataIndex2bboxIndex,
        rep_mode=coeff_tensor_args["rep_mode"],
        colspan_coeff_mode=coeff_tensor_args["colspan_coeff_mode"],
    )
    coeff_tensor = torch.stack(
        [
            rowwise_coeff_tensor,
            colwise_coeff_tensor,
        ],
        dim=0,
    )  # shape: (2, bbox_token_cnt, bbox_token_cnt)

    # 4. Shift pointer_label & is_data_tensor as causality is considered (i.e. shift for <thead>)
    pointer_label = pointer_label[
        1:
    ]  # (seq_length - bbox_token_cnt - 2, bbox_token_cnt)
    is_data_tensor = is_data_tensor[1:]  # (seq_length - bbox_token_cnt - 2)

    return (
        pointer_label,
        is_data_tensor,
        new_filtered_dr_coords,
        new_filtered_texts,
        coeff_tensor,
    )


def get_cell_pointer_label(
    gold_cell_coords: List,
    gold_coord_isFilled: List[bool],
    gold_text: List[str],
    cell_shuffle_rate: float,
    input_ids: torch.IntTensor,
    data_ids: List[int],
    bbox_token_cnt: int,
    coeff_tensor_args: dict,
):
    """
    This function generates the pointer label when using cell-level information (i.e. gold cell coords).

    Args:
        gold_cell_coords list(list(int)): list of gold cell coords, each cell coord is a list of 4 int (x1, y1, x2, y2)
        gold_coord_isFilled list(bool): list of bool indicating if <td> is filled with data
        gold_text list(str): list of str containing text in <td>
        cell_shuffle_rate float: rate of shuffling cells (for training mode only)
        input_ids torch.IntTensor: input_ids of shape (seq_length)
        data_ids list(int): list of token_ids which correspond to data tokens (e.g. <td> for html)
        bbox_token_cnt int: number of tokens in sequence dedicated for bbox coord representation (e.g. 768)

    """
    # shared variable set-up
    seq_length = input_ids.shape[0]

    # 1. First get bool tensor indicating which of the input_ids correspond to data_ids
    is_data_tensor = torch.zeros(
        (seq_length - bbox_token_cnt - 1), dtype=torch.bool
    )  # -1 for bos token (causal supervision)
    for data_id in data_ids:
        is_data_tensor = torch.logical_or(
            is_data_tensor, input_ids[1:-bbox_token_cnt] == data_id
        )

    # 2. Processing gold cell information
    # 2.1 First filter out all gold_cell_coords which are not filled -- to make subsequent steps easier
    filtered_gold_cell_coords = [
        x for i, x in enumerate(gold_cell_coords) if gold_coord_isFilled[i]
    ]
    filtered_gold_cell_texts = [
        x for i, x in enumerate(gold_text) if gold_coord_isFilled[i]
    ]

    filtered_org_index_list = [
        i for i, x in enumerate(gold_coord_isFilled) if x
    ]  # recently added

    # 2.2 Serialize gold_cell_coords from top-left to bottom-right
    filtered_gold_cell_coords, filtered_gold_cell_texts, serialized_org_index_list = (
        serialize_bbox_top_left_bottom_right(
            filtered_gold_cell_coords, filtered_gold_cell_texts
        )
    )
    filtered_org_index_list = [
        filtered_org_index_list[i] for i in serialized_org_index_list
    ]

    # 2.3 In event that no. of gold cells exceed bbox_token_cnt, slice it. NOTE -1 as first bbox is used for empty cell pointing
    if len(filtered_gold_cell_coords) > (bbox_token_cnt - 1):
        filtered_gold_cell_coords = filtered_gold_cell_coords[: bbox_token_cnt - 1]
        filtered_gold_cell_texts = filtered_gold_cell_texts[: bbox_token_cnt - 1]
        filtered_org_index_list = filtered_org_index_list[: bbox_token_cnt - 1]

    # 2.4 Shuffle gold_cell_coords at cell_shuffle_rate
    if random.random() < cell_shuffle_rate:
        zipped_tmp = list(
            zip(
                filtered_org_index_list,
                filtered_gold_cell_coords,
                filtered_gold_cell_texts,
            )
        )
        random.shuffle(zipped_tmp)
        new_gold_cell_indices, new_gold_cell_coords, new_gold_cell_texts = zip(
            *zipped_tmp
        )
    else:
        new_gold_cell_indices = filtered_org_index_list
        new_gold_cell_coords = filtered_gold_cell_coords
        new_gold_cell_texts = filtered_gold_cell_texts

    # 3. Get pointer label
    pointer_label = []
    dataIndex2bboxIndex = {}
    data_cell_index = 0
    for i in range(is_data_tensor.shape[0]):  # iter over (seq_len - bbox_token_cnt - 1)
        tmp_label = torch.zeros((bbox_token_cnt), dtype=input_ids.dtype)  # seq of 0

        if is_data_tensor[
            i
        ]:  # this is a data token, i.e. either '<td>' or '<td ' for HTML and C-tag for OTSL
            # Get the next valid data_cell_index

            if gold_coord_isFilled[
                data_cell_index
            ]:  # this cell is supposed to be filled with content
                if data_cell_index in new_gold_cell_indices:
                    tmp_indx = new_gold_cell_indices.index(data_cell_index)
                    tmp_label[tmp_indx + 1] = 1
                    dataIndex2bboxIndex[i] = [tmp_indx]
                else:
                    # corresponding bbox missing due to exceeding bbox_token_cnt
                    is_data_tensor[i] = (
                        False  # set to False so that this cell is not counted as data token
                    )
            else:
                # This is a table data with no content
                tmp_label[0] = 1

            data_cell_index += 1

        pointer_label.append(tmp_label)
    pointer_label = torch.stack(
        pointer_label, dim=0
    )  # shape: (seq_length - bbox_token_cnt - 1, bbox_token_cnt)

    # Coefficient tensor for HiMulConET
    rowwise_coeff_tensor = get_rowwise_HiMulConET_Coeff(
        sliced_input_ids=input_ids[1:-bbox_token_cnt],
        tokenizer=coeff_tensor_args["tokenizer"],
        bbox_token_cnt=bbox_token_cnt,
        is_data_tensor=is_data_tensor,
        tag2coord_map=dataIndex2bboxIndex,
        rep_mode=coeff_tensor_args["rep_mode"],
        rowspan_coeff_mode=coeff_tensor_args["rowspan_coeff_mode"],
    )
    colwise_coeff_tensor = get_columnwise_HiMulConET_Coeff(
        sliced_input_ids=input_ids[1:-bbox_token_cnt],
        tokenizer=coeff_tensor_args["tokenizer"],
        bbox_token_cnt=bbox_token_cnt,
        is_data_tensor=is_data_tensor,
        tag2coord_map=dataIndex2bboxIndex,
        rep_mode=coeff_tensor_args["rep_mode"],
        colspan_coeff_mode=coeff_tensor_args["colspan_coeff_mode"],
    )

    coeff_tensor = torch.stack(
        [
            rowwise_coeff_tensor,
            colwise_coeff_tensor,
        ],
        dim=0,
    )  # shape: (2, bbox_token_cnt, bbox_token_cnt)

    # 4. Shift pointer_label & is_data_tensor as causality is considered (i.e. shift for <thead>)
    pointer_label = pointer_label[
        1:
    ]  # (seq_length - bbox_token_cnt - 2, bbox_token_cnt)
    is_data_tensor = is_data_tensor[1:]  # (seq_length - bbox_token_cnt - 2)

    return (
        pointer_label,
        is_data_tensor,
        new_gold_cell_coords,
        new_gold_cell_texts,
        coeff_tensor,
    )


def serialize_bbox_top_left_bottom_right(coord_list, cell_text_list):
    """Serialize bounding boxes from top-left to bottom-right.

    Args:
        coord_list list(list(int)): list of bbox coords, each bbox coord is a list of 4 int (x1, y1, x2, y2)
        cell_text_list list(str): list of str containing text in bbox

    NOTE:
        Both coord_list and cell_text_list are pertaining to FILLED cells ONLY.

    Returns:
        serialized_coord_list list(list(int)): list of bbox coords, each bbox coord is a list of 4 int (x1, y1, x2, y2)
        serialized_cell_text_list list(str): list of str containing text in bbox
        serialized_org_index_list list(int): list of index of bbox in original coord_list
    """
    # 1. Get minimum height across bboxes
    min_height = min(
        [abs(x[1] - x[3]) for x in coord_list]
    )  # TODO Change to median height to see if it works better
    org_coord_indices = list(range(len(coord_list)))

    # 2. Group up bboxes by row
    row_groups = {}
    for coord, text, org_index in zip(coord_list, cell_text_list, org_coord_indices):
        top_height = coord[1]
        row_index = int(top_height // min_height)
        if row_index not in row_groups:
            row_groups[row_index] = []
        row_groups[row_index].append((coord, text, org_index))

    # 3. Sort each row by x1
    for row_index in row_groups:
        row_groups[row_index] = sorted(row_groups[row_index], key=lambda x: x[0][0])

    # 4. Output serialized coord_list and cell_text_list
    serialized_coord_list = []
    serialized_cell_text_list = []
    serialized_org_index_list = []
    for row_index in sorted(row_groups.keys()):
        for coord, text, org_index in row_groups[row_index]:
            serialized_coord_list.append(coord)
            serialized_cell_text_list.append(text)
            serialized_org_index_list.append(org_index)

    return (
        serialized_coord_list,
        serialized_cell_text_list,
        serialized_org_index_list,
    )


def get_bbox_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1 (Tuple): Tuple of 4 float values (x1, y1, x2, y2).
        bbox2 (Tuple): Tuple of 4 float values (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    x5 = max(x1, x3)
    y5 = max(y1, y3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)

    if x5 >= x6 or y5 >= y6:
        return 0

    intersection_area = (x6 - x5) * (y6 - y5)
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (x4 - x3) * (y4 - y3)

    return intersection_area / (bbox1_area + bbox2_area - intersection_area + 1e-6)

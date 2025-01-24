import argparse
import json
import pickle
import random

import torch
from tqdm import tqdm

from .preprocess_data_utils import convert_html_to_otsl


def format_pubtabnet_gold_coords(gold_bbox_collection):
    """Preprocess gold coordinate info for PubTabNet dataset.

    NOTE
        - In PubTabnet, empty cells are marked by absence of 'bbox' in the cell's dictionary

    Args:
        gold_bbox_collection List[Dict]: List of cell dictionaries
            Each dictionary has 'tokens' and 'bbox' (if the cell is filled) keys
            E.g. [{'tokens': ['<b>', 'R', 'i', 's', 'k', ' ', 'F', 'a', 'c', 't', 'o', 'r', 's', '</b>'], 'bbox': [28, 5, 77, 14]}, ... ]
    """
    cells = []
    for cell in gold_bbox_collection:
        if "bbox" in cell:
            # This is a cell with filledContent
            string_coords = ["%.2f" % c for c in cell["bbox"]] + ["2"]
            # Add serialised string
            text = "".join(cell["tokens"])
            string_coords = string_coords + [text]
            cells.append(" ".join(string_coords))
        else:
            # This is an empty cell
            string_coords = ["-1.0", "-1.0", "-1.0", "-1.0", "1"]
            text = ""
            string_coords = string_coords + [text]
            cells.append(" ".join(string_coords))

    return cells


def group_det_bbox(
    pred_bbox_tensor, gold_bbox_tensor, IOU_threshold=0.1, IOP_threshold=0.1
):
    """Map pred bbox to gold bbox based on IOU.

    Args:
        pred_bbox_tensor: torch.Tensor, (N, 4)
        gold_bbox_tensor: torch.Tensor, (M, 4)
        IOU_threshold: float, threshold for IOU
        IOP_threshold: float, threshold for IOP

    """

    x_left_y_top_tensor = torch.max(
        pred_bbox_tensor.unsqueeze(1)[:, :, :2], gold_bbox_tensor.unsqueeze(0)[:, :, :2]
    )  # (N, M, 2)
    x_right_y_bottom_tensor = torch.min(
        pred_bbox_tensor.unsqueeze(1)[:, :, 2:], gold_bbox_tensor.unsqueeze(0)[:, :, 2:]
    )  # (N, M, 2)

    x_left = x_left_y_top_tensor[:, :, 0]
    y_top = x_left_y_top_tensor[:, :, 1]
    x_right = x_right_y_bottom_tensor[:, :, 0]
    y_bottom = x_right_y_bottom_tensor[:, :, 1]

    # Compute the intersection area
    intersection_area = torch.logical_or(x_right < x_left, y_bottom < y_top).float()
    intersection_area = 1 - intersection_area  # (N, M)
    intersection_area = intersection_area * (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox1_area = (pred_bbox_tensor[:, 2] - pred_bbox_tensor[:, 0]) * (
        pred_bbox_tensor[:, 3] - pred_bbox_tensor[:, 1]
    )  # (N,)
    bbox2_area = (gold_bbox_tensor[:, 2] - gold_bbox_tensor[:, 0]) * (
        gold_bbox_tensor[:, 3] - gold_bbox_tensor[:, 1]
    )  # (M,)

    # Compute the IOU
    iou = intersection_area / (
        bbox1_area.unsqueeze(1) + bbox2_area.unsqueeze(0) - intersection_area + 1e-6
    )  # (N, M)

    # Map the pred bbox to gold bbox
    iou_max, iou_gold_bbox_idx = torch.max(iou, dim=1)  # (N,)
    iou_pred_bbox_idx = torch.arange(pred_bbox_tensor.shape[0])  # (N,)
    iou_pred_bbox_idx = iou_pred_bbox_idx[iou_max > IOU_threshold]
    iou_gold_bbox_idx = iou_gold_bbox_idx[iou_max > IOU_threshold]

    # For preds not associated with any gold bbox, recheck and associate if the overlap of pred bbox is > 0.1
    iop = intersection_area / bbox1_area.unsqueeze(1)  # (N, M)
    iop_max, iop_gold_bbox_idx = torch.max(iop, dim=1)  # (N,)
    iop_pred_bbox_idx = torch.arange(pred_bbox_tensor.shape[0])  # (N,)
    bool_mask = torch.logical_and(iop_max > IOP_threshold, iou_max <= IOU_threshold)
    iop_pred_bbox_idx = iop_pred_bbox_idx[bool_mask]
    iop_gold_bbox_idx = iop_gold_bbox_idx[bool_mask]

    pred_bbox_idx = torch.cat([iou_pred_bbox_idx, iop_pred_bbox_idx], dim=0)
    gold_bbox_idx = torch.cat([iou_gold_bbox_idx, iop_gold_bbox_idx], dim=0)

    return pred_bbox_idx, gold_bbox_idx, iou, intersection_area


def preprocess_det_bbox(
    pred_bbox_collection, gold_bbox_collection, IOU_threshold=0.1, IOP_threshold=0.1
):
    """Preprocess detected bbox and gold bbox.

    Args:
        pred_bbox_collection: List[List[float]], list of detected bounding boxes
            Each detected bounding box is represented as [x1, y1, x2, y2, x3, y3, x4, y4]
        gold_bbox_collection: List[Dict], list of gold bounding boxes
        IOU_threshold: float, threshold for IOU (Intersection over Union)
        IOP_threshold: float, threshold for IOP (Intersection over Prediction)
    """

    # Reformat bounding boxes to [x_left, y_top, x_right, y_bottom]
    pred_cell_bboxes = [
        [
            min(coord[0], coord[2], coord[4], coord[6]),
            min(coord[1], coord[3], coord[5], coord[7]),
            max(coord[0], coord[2], coord[4], coord[6]),
            max(coord[1], coord[3], coord[5], coord[7]),
        ]
        for coord in pred_bbox_collection
    ]

    gold_cell_bboxes = [x["bbox"] for x in gold_bbox_collection if "bbox" in x]
    gold_cell_contents = [
        "".join(x["tokens"]) for x in gold_bbox_collection if "bbox" in x
    ]
    grp_to_filled_gold_idx_mapping = {}
    current_filled_idx = 0
    for gold_idx, gold_bbox in enumerate(gold_bbox_collection):
        if "bbox" in gold_bbox:
            grp_to_filled_gold_idx_mapping[current_filled_idx] = gold_idx
            current_filled_idx += 1

    pred_bbox_tensor = torch.tensor(pred_cell_bboxes)
    gold_bbox_tensor = torch.tensor(gold_cell_bboxes)

    pred_bbox_idx, gold_bbox_idx, iou, intersection_area = group_det_bbox(
        pred_bbox_tensor,
        gold_bbox_tensor,
        IOU_threshold=IOU_threshold,
        IOP_threshold=IOP_threshold,
    )

    # Group up pred_idx by common gold_bbox_idx
    pred_bbox_idx_group = {}
    for pred_idx, gold_idx in zip(pred_bbox_idx, gold_bbox_idx):
        if gold_idx.item() not in pred_bbox_idx_group:
            pred_bbox_idx_group[gold_idx.item()] = []
        pred_bbox_idx_group[gold_idx.item()].append(pred_idx.item())

    def sort_bbox_coords(bbox_coord_list):
        if len(bbox_coord_list) == 1:
            return bbox_coord_list
        else:
            new_list = []
            # 1. get min_height across all bboxes in the list
            min_height = 100000
            for bbox_coord in bbox_coord_list:
                min_height = min(min_height, bbox_coord[3] - bbox_coord[1])

            sorted_by_height_interval = {}
            for bbox_coord in bbox_coord_list:
                height_interval = int(bbox_coord[1] / min_height)
                if height_interval not in sorted_by_height_interval:
                    sorted_by_height_interval[height_interval] = []
                sorted_by_height_interval[height_interval].append(bbox_coord)

            for height_interval in sorted(sorted_by_height_interval.keys()):
                bbox_coords = sorted_by_height_interval[height_interval]
                # sort bbox_coords by x_left
                bbox_coords = sorted(bbox_coords, key=lambda x: x[0])
                new_list.extend(bbox_coords)

            return new_list

    # First, serialize bbox coords within each group
    for gold_idx in pred_bbox_idx_group.keys():
        pred_bbox_idx_group[gold_idx] = sort_bbox_coords(
            [pred_cell_bboxes[x] for x in pred_bbox_idx_group[gold_idx]]
        )

    # Next, serialize groups
    gold_idx_first_bbox_list = [(k, v[0]) for k, v in pred_bbox_idx_group.items()]
    minimum_group_height = 100000
    for gold_idx, bbox_coord in gold_idx_first_bbox_list:
        minimum_group_height = min(minimum_group_height, bbox_coord[3] - bbox_coord[1])

    sorted_by_group_height_interval = {}
    for gold_idx, bbox_coord in gold_idx_first_bbox_list:
        height_interval = int(bbox_coord[1] / minimum_group_height)
        if height_interval not in sorted_by_group_height_interval:
            sorted_by_group_height_interval[height_interval] = []
        sorted_by_group_height_interval[height_interval].append((gold_idx, bbox_coord))

    sorted_gold_idx_first_bbox_list = []
    for height_interval in sorted(sorted_by_group_height_interval.keys()):
        gold_idx_first_bbox_list = sorted_by_group_height_interval[height_interval]
        # sort bbox_coords by x_left
        gold_idx_first_bbox_list = sorted(
            gold_idx_first_bbox_list, key=lambda x: x[1][0]
        )
        sorted_gold_idx_first_bbox_list.extend(gold_idx_first_bbox_list)

    # Finally, serialize the whole list
    serialized_pred_bbox = {}
    for group_idx, group_data in enumerate(sorted_gold_idx_first_bbox_list):
        serialized_pred_bbox[group_idx] = [
            pred_bbox_idx_group[group_data[0]],
            grp_to_filled_gold_idx_mapping[group_data[0]],
            gold_cell_contents[group_data[0]],
        ]

    return serialized_pred_bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_config_path", type=str, help="Path to the data config file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to save the preprocessed data"
    )
    parser.add_argument(
        "--bin_idx", type=int, default=-1, help="Index of the bin to process"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=0,
        help="Number of bins to split the dataset into",
    )
    args = parser.parse_args()

    # sanity check
    if args.bin_idx != -1:
        assert args.num_bins > 0

    # config loading and setting up for preprocessing
    data_config = json.load(open(args.data_config_path, "r"))
    random.seed(42)
    generated_dataset = {
        k: [] for k in data_config["SPLITS"]
    }  # This is to store all preprocessed data

    # Data loading
    meta_pubtabnet_data = open(data_config["PUBTABNET_PATH"], "r").readlines()
    pse_det_data = {"train": {}, "validation": {}}
    for split in data_config["SPLITS"]:
        for pickle_path in tqdm(
            data_config["DR_COORD_PATH"][split].values(),
            desc="Loading PSE Det result for %s" % split,
        ):
            pse_data_list_loaded = pickle.load(open(pickle_path, "rb"))
            for pse_data in pse_data_list_loaded:
                pse_det_data[split][pse_data["file_name"]] = pse_data

    # In PubTabNet, ambiguous HTML representations for training and validation datasets are removed. NOTE: this is not done for test dataset
    split_dataset = {"train": [], "validation": []}
    ambiguous_data_filenames = []
    for split_type, split_amb_filenames in json.load(
        open(data_config["AMBIGUOUS_DATA_PATH"], "r")
    ).items():
        ambiguous_data_filenames.extend(split_amb_filenames)
    for raw_data in tqdm(meta_pubtabnet_data, desc="Removing amgiguous data"):
        data = json.loads(raw_data)
        if data["filename"] in ambiguous_data_filenames:
            continue
        split = data["split"]
        if split == "val":
            split = "validation"
        assert split in ["train", "validation"], (
            "Invalid split %s" % split
        )  # NOTE: Test dataset is not processed at this stage
        split_dataset[split].append(raw_data)

    # Preprocessing
    # Preprocess Train and Validation dataset
    for split in ["train", "validation"]:
        if args.bin_idx != -1:
            # Split the dataset into bins
            num_data = len(split_dataset[split])
            bin_size = int(num_data / args.num_bins)
            start_idx = args.bin_idx * bin_size
            if args.bin_idx == args.num_bins - 1:
                end_idx = num_data
            else:
                end_idx = start_idx + bin_size
            sliced_dataset = split_dataset[split][start_idx:end_idx]
        else:
            sliced_dataset = split_dataset[split]

        for raw_data in tqdm(sliced_dataset, desc="Pre-processing split %s" % split):
            loaded_data = json.loads(raw_data)
            data_filename = loaded_data["filename"]

            # GET OTSL representation
            otsl_seq, num_rows, num_cols = convert_html_to_otsl(
                html_seq=loaded_data["html"]["structure"]["tokens"],
                otsl_tag_maps=data_config["OTSL_TAG"],
            )
            gold_bbox_seq = format_pubtabnet_gold_coords(loaded_data["html"]["cells"])

            # Get pse det result
            pse_det_result = pse_det_data[split][data_filename]

            # Get pred and gold bbox idx grouping
            # pred_bbox_idx is a dict mapping group_idx to list of bbox_coords
            pred_bbox_idx = preprocess_det_bbox(
                pse_det_result["bbox"],
                loaded_data["html"]["cells"],
                IOU_threshold=0.1,
                IOP_threshold=0.1,
            )

            data_entry = {
                "file_name": loaded_data["filename"],
                "dr_coord": pred_bbox_idx,
                "gold_coord": gold_bbox_seq,
                "org_html": loaded_data["html"]["structure"]["tokens"],
                "otsl_seq": otsl_seq,
                "num_rows": num_rows,
                "num_cols": num_cols,
                "split": split,
            }

            generated_dataset[split].append(data_entry)

        list_of_data = generated_dataset[split]
        if args.bin_idx != -1:
            savename = "%s/dataset_%s_%d_%d.jsonl" % (
                args.output_dir,
                split,
                args.bin_idx,
                args.num_bins,
            )
        else:
            savename = "%s/dataset_%s.jsonl" % (args.output_dir, split)
        with open(savename, "w") as f:
            for d in list_of_data:
                f.write(json.dumps(d) + "\n")

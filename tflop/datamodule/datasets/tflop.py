import json
import os
import pickle

from PIL import Image
from omegaconf import DictConfig
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from tflop.datamodule.preprocess.common_utils import (
    convert_gold_coords,
    generate_filled_html,
    get_cell_pointer_label,
    get_dr_pointer_label,
    int_convert_and_pad_coords,
    rescale_bbox,
    serialize_bbox_top_left_bottom_right,
)
from tflop.datamodule.preprocess.image_utils import prepare_image_tensor


class TFLOPDataset(Dataset):
    def __init__(
        self: "TFLOPDataset",
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        config: DictConfig = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.config = config

        # Run sanity check on implementation status to avoid unexpected runs / errors
        self.implementation_check()

        # Set up necessary variables
        self.set_up_variables()

        # Load dataset
        self.manual_load_dataset()

        # Set up necessary token ids
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

        # Initialize OTSL config
        self.initialize_OTSL_config()

    def implementation_check(self: "TFLOPDataset"):
        """
        Sanity check of implementation status to avoid unexpected runs / behaviour
        """
        if self.split not in ["train", "validation"]:
            raise ValueError(
                "split must be either 'train' or 'validation'. But got %s" % self.split
            )

        # Shuffling of bbox coordinates should only be done in training set
        if self.config.get("shuffle_cell_bbox_rate", 0.0) != 0.0:
            assert (
                self.split == "training"
            ), "shuffle_cell_bbox_rate must be 0.0 when split is not 'training'."

        # OTSL config check
        if not self.config.get("use_OTSL", False):
            raise NotImplementedError("Non OTSL modes are deprecated.")

    def set_up_variables(self: "TFLOPDataset"):
        # General
        self.image_path = self.config.get("image_path", None)
        self.meta_data_path = self.config.get("meta_data_path", None)
        self.prompt_end_token = self.config.get("prompt_end_token", "<s_answer>")
        self.input_img_size = (
            self.config.input_size.width,
            self.config.input_size.height,
        )

        # Pointer-specific
        self.bbox_token_cnt = self.config.get("bbox_token_cnt", None)
        self.max_length = self.config.get("max_length", None)
        self.use_cell_bbox = self.config.get("use_cell_bbox", False)
        self.shuffle_cell_bbox_rate = self.config.get("shuffle_cell_bbox_rate", 0.0)

        # Experimental
        self.use_OTSL = True
        self.span_coeff_mode = self.config.get("span_coeff_mode", "proportional")

        self.coeff_tensor_args = {
            "tokenizer": None,
            "rep_mode": None,
            "rowspan_coeff_mode": self.span_coeff_mode,
            "colspan_coeff_mode": self.span_coeff_mode,
        }

        self.coeff_tensor_args["rep_mode"] = "OTSL"

    def manual_load_dataset(self: "TFLOPDataset"):
        """
        Load necessary data for experiment.
        """
        meta_data_path = os.path.join(
            self.meta_data_path,
            "dataset_%s.jsonl" % self.split
        )
        self.metadata_loaded = open(meta_data_path).readlines()

    def initialize_OTSL_config(self: "TFLOPDataset"):
        """Initialize tokens which signify data cell in OTSL representations."""

        if self.use_OTSL:
            self.data_ids = [
                self.tokenizer.convert_tokens_to_ids(token) for token in ["C-tag"]
            ]
        else:
            raise NotImplementedError("Non OTSL modes are deprecated.")

    def __len__(self: "TFLOPDataset") -> int:
        return len(self.metadata_loaded)

    def __getitem__(self: "TFLOPDataset", idx: int):
        """
        Get a sample from the dataset for DonutPointer.
        """
        data_selected = self.load_sampled_metadata(idx)
        sample = self.build_sample_dictionary(data_selected)

        # Get Model input sequence and token ids
        input_parser_seq, input_parse, input_ids = self.get_input_parser_seq_and_tok(
            sample
        )

        # Load and prepare image as tensor
        img_tensor, org_img_size, padding_dims = self.load_and_prepare_img_tensor(
            sample
        )

        # Process gold coord information
        gold_coords = convert_gold_coords(sample["gold_coord"])

        # html_with_content: html seq with content for edit-distance evaluation against pred seq later (i.e. for evaluation)
        html_with_content = generate_filled_html(
            gold_text_list=gold_coords["text"],
            is_cell_filled=gold_coords["isFilled"],
            org_html_list=sample["org_html"],
        )

        self.coeff_tensor_args["tokenizer"] = self.tokenizer

        if self.use_cell_bbox:
            # 1. Rescale DR coordinates / or gold coords if self.use_cell_bbox is True
            rescaled_coords = self.rescale_gold_coordinates(
                gold_coords, org_img_size, padding_dims
            )

            # 2. Get pointer & pointer_mask labels
            gold_coord_info = {
                "isFilled": (gold_coords["isFilled"]),
                "text": (gold_coords["text"]),
            }
            (
                pointer_label,
                pointer_mask_label,
                rescaled_coords,
                cell_texts,
                bbox_coeff_tensor,
            ) = get_cell_pointer_label(
                gold_cell_coords=rescaled_coords,
                gold_coord_isFilled=gold_coord_info["isFilled"],
                gold_text=gold_coord_info["text"],
                cell_shuffle_rate=self.shuffle_cell_bbox_rate,
                input_ids=input_ids,
                data_ids=self.data_ids,
                bbox_token_cnt=self.bbox_token_cnt,
                coeff_tensor_args=self.coeff_tensor_args,
            )

        else:
            (
                pointer_label,
                pointer_mask_label,
                rescaled_coords,
                cell_texts,
                bbox_coeff_tensor,
            ) = get_dr_pointer_label(
                dr_coords=sample["dr_coord"],
                gold_coord_isFilled=gold_coords["isFilled"],
                cell_shuffle_rate=self.shuffle_cell_bbox_rate,
                input_ids=input_ids,
                data_ids=self.data_ids,
                bbox_token_cnt=self.bbox_token_cnt,
                org_img_size=org_img_size,
                new_img_size=self.input_img_size,
                padding_dims=padding_dims,
                coeff_tensor_args=self.coeff_tensor_args,
            )

        # 3. Convert to int and pad to bbox_token_cnt
        padding_coord = [
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
        ]
        coords_int_padded = int_convert_and_pad_coords(
            coords=rescaled_coords,
            padding_coord=padding_coord,
            max_length=self.bbox_token_cnt,
        )

        # 4. Convert int padded coords into tensor
        valid_coord_length = torch.tensor(len(rescaled_coords))  # (1)
        pointer_label = pointer_label.to(torch.bool)

        if any(
            [self.config.use_RowWise_contLearning, self.config.use_ColWise_contLearning]
        ):
            chosen_bbox_coeff_tensor = []
            if self.config.use_RowWise_contLearning:
                chosen_bbox_coeff_tensor.append(bbox_coeff_tensor[0])
            if self.config.use_ColWise_contLearning:
                chosen_bbox_coeff_tensor.append(bbox_coeff_tensor[1])
            chosen_bbox_coeff_tensor = torch.stack(chosen_bbox_coeff_tensor, dim=0)
        else:
            chosen_bbox_coeff_tensor = torch.zeros_like(bbox_coeff_tensor[0])
            chosen_bbox_coeff_tensor = chosen_bbox_coeff_tensor.unsqueeze(0)

        # Slice input_ids to max_length - bbox_token_length, i.e. txt_length
        input_ids = input_ids[: self.max_length - self.bbox_token_cnt]

        if self.split == "train":
            token_pred_labels = input_ids.clone()
            token_pred_labels[token_pred_labels == self.tokenizer.pad_token_id] = -100
            token_pred_labels[
                : torch.nonzero(token_pred_labels == self.prompt_end_token_id).sum() + 1
            ] = -100

            return (
                img_tensor,
                input_ids,
                coords_int_padded,
                valid_coord_length,
                token_pred_labels,
                pointer_label,
                pointer_mask_label,
                chosen_bbox_coeff_tensor,
            )
        else:
            # This is for validation set

            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()

            # Join up cell_texts into a single string for collation
            cell_text_collated = "<special_cell_text_sep>".join(cell_texts)

            return (
                img_tensor,
                input_ids,
                coords_int_padded,
                valid_coord_length,
                prompt_end_index,
                input_parse,
                pointer_label,
                pointer_mask_label,
                html_with_content,
                cell_text_collated,
                sample["file_name"],
                chosen_bbox_coeff_tensor,
            )

    def load_sampled_metadata(self, idx):
        """Given data idx, load corresponding metadata info."""

        sampled_metadata = json.loads(self.metadata_loaded[idx])
        return sampled_metadata

    def build_sample_dictionary(self, data_selected):
        """Build sampled data dictionary."""
        sample = {}
        image_path = os.path.join(
            self.image_path,
            self.split,
            data_selected["file_name"]
        )
        img = Image.open(image_path)
        sample["image"] = img
        for data_k, data_v in data_selected.items():
            sample[data_k] = data_v
        return sample

    def get_input_parser_seq_and_tok(self, sample):
        """Get input text sequence and tokens for model."""
        # 1. OTSL sequence
        OTSL_data = "".join(sample["otsl_seq"]) if self.use_OTSL else None

        # 2. tokenize
        input_parse = (
            self.tokenizer.bos_token
            + self.prompt_end_token
            + OTSL_data
            + "</s_answer>"
            + self.tokenizer.eos_token
        )
        input_ids = self.tokenizer(
            input_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return OTSL_data, input_parse, input_ids

    def load_and_prepare_img_tensor(self, sample):
        """Load and prepare image as tensor for model."""
        img_tensor, org_img_size, padding_dims = prepare_image_tensor(
            input_image=sample["image"],
            target_img_size=self.input_img_size,  # (width, height)
            random_padding=self.split == "train",
        )

        return img_tensor, org_img_size, padding_dims

    def rescale_gold_coordinates(self, gold_coords, org_img_size, padding_dims):
        """Rescale gold coordinates."""
        rescale_fn, coord_input = rescale_bbox, gold_coords["coords"]
        rescaled_coords = rescale_fn(
            list_of_coords=coord_input,
            org_img_size=org_img_size,
            new_img_size=self.input_img_size,
            padding_dims=padding_dims,
        )

        return rescaled_coords


class TFLOPTestDataset(Dataset):
    """Simplified test dataset class for TFLOP."""

    def __init__(
        self: "TFLOPTestDataset",
        tokenizer: PreTrainedTokenizer,
        split: str = "test",
        config: DictConfig = None,
        aux_json_path: str = None,
        aux_img_path: str = None,
        aux_rec_pkl_path: str = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.config = config
        self.aux_json_path = aux_json_path
        self.aux_img_path = aux_img_path
        self.aux_rec_pkl_path = aux_rec_pkl_path

        # Run sanity check on implementation status to avoid unexpected runs / errors
        self.implementation_check()

        # Set up necessary variables
        self.set_up_variables()

        # Load dataset
        self.manual_load_dataset()

        # Set up necessary token ids
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.prompt_end_token
        )

        # Initialize OTSL config
        self.initialize_OTSL_config()

    def implementation_check(self: "TFLOPTestDataset"):
        """
        Sanity check of implementation status to avoid unexpected runs / behaviour
        """
        if self.split not in ["validation", "test"]:
            raise ValueError(
                "split must be either validation or test. But got %s" % self.split
            )

        # Shuffling of bbox coordinates should only be done in training set
        assert (
            self.config.get("shuffle_cell_bbox_rate", 0.0) == 0.0
        ), "shuffle_cell_bbox_rate must be 0.0 when not training."

    def set_up_variables(self: "TFLOPTestDataset"):
        # General
        self.prompt_end_token = self.config.get("prompt_end_token", "<s_answer>")
        self.input_img_size = (
            self.config.input_size.width,
            self.config.input_size.height,
        )

        # Pointer-specific
        self.bbox_token_cnt = self.config.get("bbox_token_cnt", None)
        self.max_length = self.config.get("max_length", None)
        self.use_cell_bbox = self.config.get("use_cell_bbox", False)
        self.shuffle_cell_bbox_rate = self.config.get("shuffle_cell_bbox_rate", 0.0)

        # Experimental
        self.use_OTSL = self.config.get("use_OTSL", False)

    def manual_load_dataset(self: "TFLOPTestDataset"):
        """
        Load dataset for evaluation.
        NOTE:
            - If metadata.jsonl path is present, load from there.
            - Else, load from aux_json_path.
        """
        for aux_filepath in [self.aux_json_path, self.aux_rec_pkl_path]:
            assert (
                aux_filepath is not None
            ), "aux_json_path and aux_rec_pkl_path must be provided."
            assert os.path.exists(aux_filepath), (
                "aux_filepath %s does not exist." % aux_filepath
            )
        assert os.path.exists(self.aux_img_path), (
            "aux_img_path %s does not exist." % self.aux_img_path
        )
        self.using_metadata_jsonl = False

        self.aux_json = json.load(open(self.aux_json_path, "r"))
        self.aux_rec = pickle.load(open(self.aux_rec_pkl_path, "rb"))

    def initialize_OTSL_config(self: "TFLOPTestDataset"):
        """Initialize tokens which signify data cell in OTSL representations."""

        if self.use_OTSL:
            self.data_ids = [
                self.tokenizer.convert_tokens_to_ids(token) for token in ["C-tag"]
            ]
        else:
            raise NotImplementedError("Non OTSL modes are deprecated.")

    def __len__(self: "TFLOPTestDataset") -> int:

        if self.using_metadata_jsonl:
            return len(self.metadata_loaded)
        else:
            return len(self.aux_json)

    def __getitem__(self: "TFLOPTestDataset", idx: int):
        """
        Get a sample from the dataset for evaluation.
        """
        # Loading sample data
        sample = self.load_aux_data(idx)

        # Prepare image tensor
        img_tensor, org_img_size, padding_dims = prepare_image_tensor(
            input_image=sample["image"],
            target_img_size=self.input_img_size,  # (width, height)
            random_padding=False,
        )

        # Get full html with content & cell-wise coord & cell-wise text
        html_with_content, rescaled_coords, cell_texts = (
            self.get_coord_and_html_with_content(sample, org_img_size, padding_dims)
        )
        padding_coord = [
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
            self.input_img_size[0] + 3,
            self.input_img_size[1] + 3,
        ]
        coords_int_padded = int_convert_and_pad_coords(
            coords=rescaled_coords,
            padding_coord=padding_coord,
            max_length=self.bbox_token_cnt,
        )
        valid_coord_length = torch.tensor(len(rescaled_coords))  # (1)

        # Prepare input token ids
        input_parse = self.tokenizer.bos_token + self.prompt_end_token
        input_ids = self.tokenizer(
            input_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        input_ids = input_ids[: self.max_length - self.bbox_token_cnt]

        # Return values
        prompt_end_index = torch.nonzero(input_ids == self.prompt_end_token_id).sum()
        cell_text_collated = "<special_cell_text_sep>".join(cell_texts)

        return (
            img_tensor,
            input_ids,
            coords_int_padded,
            valid_coord_length,
            prompt_end_index,
            html_with_content,
            cell_text_collated,
            sample["file_name"],
        )


    def load_aux_data(self: "TFLOPTestDataset", idx: int):
        """
        When not using metadata jsonl, load metadata from aux json, aux det pkl and aux rec pkl.

        Returns:
            sample: dict
                - image: PIL Image
                - full_html_seq: str, full html sequence as string
                - html_type: str, whether the html seq is simple or complex
                - bbox_coords: list of bbox lists # was previously cell_bboxes
                - bbox_texts: list of bbox texts # was previously cell_texts
                - file_name: str, image filename
        """
        assert (
            not self.using_metadata_jsonl
        ), "load_aux_data should only be called when not using metadata jsonl."

        # Get image filenames
        img_filenames = list(self.aux_json.keys())
        img_filenames.sort()
        selected_filename = img_filenames[idx]

        # 1. Load image
        sample = {
            "image": Image.open(os.path.join(self.aux_img_path, selected_filename))
        }

        # 2. Load other data
        sample["full_html_seq"] = self.aux_json[selected_filename]["html"]
        sample["html_type"] = self.aux_json[selected_filename]["type"]

        rec_result = self.aux_rec[selected_filename]
        sample["bbox_coords"] = [list(c["bbox"]) for c in rec_result]
        sample["bbox_texts"] = [c["text"] for c in rec_result]
        sample["file_name"] = selected_filename

        return sample

    def get_coord_and_html_with_content(
        self: "TFLOPTestDataset", sample, org_img_size, padding_dims
    ):
        if self.using_metadata_jsonl:
            gold_coords = convert_gold_coords(sample["gold_coord"])

            html_with_content = generate_filled_html(
                gold_text_list=gold_coords["text"],
                is_cell_filled=gold_coords["isFilled"],
                org_html_list=sample["org_html"],
            )
            if self.use_cell_bbox:
                # rescale coords
                rescaled_coords = self.rescale_gold_coordinates(
                    gold_coords, org_img_size, padding_dims
                )

                # Only retrieved filled cells
                rescaled_coords = [
                    x
                    for i, x in enumerate(rescaled_coords)
                    if gold_coords["isFilled"][i]
                ]
                cell_texts = [
                    x
                    for i, x in enumerate(gold_coords["text"])
                    if gold_coords["isFilled"][i]
                ]

                # serialize the bbox coords and texts
                rescaled_coords, cell_texts, _ = serialize_bbox_top_left_bottom_right(
                    rescaled_coords, cell_texts
                )
                if len(rescaled_coords) > (self.bbox_token_cnt - 1):
                    rescaled_coords = rescaled_coords[: (self.bbox_token_cnt - 1)]
                    cell_texts = cell_texts[: (self.bbox_token_cnt - 1)]

            else:
                filtered_dr_coords = []
                filtered_texts = []

                dr_coord_keys = [int(x) for x in list(sample["dr_coord"].keys())]
                for tmp_idx in sorted(dr_coord_keys):
                    rescaled_coords = rescale_bbox(
                        list_of_coords=sample["dr_coord"][str(tmp_idx)][0],
                        org_img_size=org_img_size,
                        new_img_size=self.input_img_size,
                        padding_dims=padding_dims,
                    )
                    filtered_dr_coords.extend(rescaled_coords)
                    filtered_texts.append(sample["dr_coord"][str(tmp_idx)][2])
                    filtered_texts += [""] * (len(rescaled_coords) - 1)

                filtered_dr_coords, filtered_texts, _ = (
                    serialize_bbox_top_left_bottom_right(
                        filtered_dr_coords, filtered_texts
                    )
                )

                if len(filtered_dr_coords) > (self.bbox_token_cnt - 1):
                    filtered_dr_coords = filtered_dr_coords[: (self.bbox_token_cnt - 1)]
                    filtered_texts = filtered_texts[: (self.bbox_token_cnt - 1)]

                rescaled_coords = filtered_dr_coords
                cell_texts = filtered_texts
        else:
            html_with_content = sample["full_html_seq"]
            rescaled_coords = rescale_bbox(
                list_of_coords=sample["bbox_coords"],
                org_img_size=org_img_size,
                new_img_size=self.input_img_size,
                padding_dims=padding_dims,
            )

            rescaled_coords, cell_texts, _ = serialize_bbox_top_left_bottom_right(
                rescaled_coords, sample["bbox_texts"]
            )
            if len(rescaled_coords) > (self.bbox_token_cnt - 1):
                rescaled_coords = rescaled_coords[: (self.bbox_token_cnt - 1)]
                cell_texts = cell_texts[: (self.bbox_token_cnt - 1)]

        return html_with_content, rescaled_coords, cell_texts

    def rescale_gold_coordinates(self, gold_coords, org_img_size, padding_dims):
        """Rescale gold coordinates."""

        rescale_fn, coord_input = rescale_bbox, gold_coords["coords"]
        rescaled_coords = rescale_fn(
            list_of_coords=coord_input,
            org_img_size=org_img_size,
            new_img_size=self.input_img_size,
            padding_dims=padding_dims,
        )

        return rescaled_coords

import os
from pathlib import Path

import omegaconf
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import torch
from transformers import AutoTokenizer


@rank_zero_only
def save_config_file(config: DictConfig, path: str) -> None:
    """
    Save a configuration file in YAML format to the specified path.

    This function takes a DictConfig object, optionally resolves its interpolations,
    and saves it as a YAML file in the given directory. If the directory does not exist,
    it will be created. The function is decorated with @rank_zero_only, which means it
    should only be executed by the process with rank 0 in a distributed setting.

    Args:
        config (DictConfig): The configuration object to be saved.
        path (str): The directory path where the configuration file should be saved.

    Returns:
        None

    Raises:
        OmegaConfException: If there is an error in resolving the interpolations.

    """
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f)
        print(f"Config is saved at {save_path}")


def set_up_tokenizer(
    pretrained_tokenizer_name_or_path, bbox_special_tokens, other_special_tokens
):
    """Set up tokenizer and add bbox & other special tokens

    Args:
        pretrained_tokenizer_name_or_path (str): pretrained tokenizer name or path
        bbox_special_tokens (ListConfig): bbox special tokens
        other_special_tokens (ListConfig): other special tokens

    Returns:
        tokenizer (AutoTokenizer): tokenizer with bbox & other special tokens added
    """
    # 1. Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name_or_path)

    # 2. Add bbox special tokens
    added_bbox_token_cnt = tokenizer.add_special_tokens(
        {"additional_special_tokens": sorted(set(bbox_special_tokens))}
    )

    # 3. Add other special tokens
    added_special_tok_cnt = tokenizer.add_special_tokens(
        {"additional_special_tokens": sorted(set(other_special_tokens))}
    )

    print(
        "Added %s bbox tokens and %s special tokens to tokenizer"
        % (added_bbox_token_cnt, added_special_tok_cnt)
    )

    return tokenizer


class ProgressBar(pl.callbacks.TQDMProgressBar):
    def __init__(self, config):
        super().__init__()
        self.enable = True
        self.config = config

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        items["exp_name"] = f"{self.config.get('exp_name', '')}"
        items["exp_version"] = f"{self.config.get('exp_version', '')}"
        return items


def set_up_logger_and_callbacks(config):
    """Set up tensorboard logger, LR & ckpt callbacks + progress bar for experiment."""
    result_path = config.result_path
    exp_name = config.exp_name
    exp_version = config.exp_version

    # 1. Setup tensorboard logger
    logger = TensorBoardLogger(
        save_dir=result_path,
        name=exp_name,
        version=exp_version,
        default_hp_metric=False,
    )

    # 2. Setup callbacks
    # 2.1 lr callback
    lr_callback = LearningRateMonitor(logging_interval="step")
    # 2.2 checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_metric",
        dirpath=Path(result_path) / exp_name / exp_version,
        filename="artifacts-{epoch:02d}-{step:07d}",
        save_top_k=1,
        save_last=True,
        mode="min",
    )

    # 3. Setup progress bar
    bar = ProgressBar(config)

    return logger, lr_callback, checkpoint_callback, bar


def set_seed(seed_value):
    """Set seed for reproducibility."""
    pytorch_lightning_version = int(pl.__version__[0])
    assert pytorch_lightning_version == 2, "Only PL version 2.x is supported."
    if pytorch_lightning_version < 2:
        pl.utilities.seed.seed_everything(seed_value, workers=True)
    else:
        import lightning_fabric

        lightning_fabric.utilities.seed.seed_everything(seed_value, workers=True)


def resolve_missing_config(model_config_dict):
    """Function to handle cases where certain config values are missing"""

    # 1. Filling in max_position_embeddings if absent while max_length is present
    if (
        "max_position_embeddings" not in model_config_dict
        and "max_length" in model_config_dict
    ):
        model_config_dict["max_position_embeddings"] = model_config_dict["max_length"]

    # 2. Set use_imgRoiAlign to False if absent
    if "use_imgRoiAlign" not in model_config_dict:
        model_config_dict["use_imgRoiAlign"] = False

    # 3. Fixing input_size format
    if type(model_config_dict["input_size"]) in [
        dict,
        omegaconf.dictconfig.DictConfig,
    ]:
        model_config_dict["input_size"] = (
            model_config_dict["input_size"]["width"],
            model_config_dict["input_size"]["height"],
        )

    return model_config_dict


def decode_OTSL_seq(otsl_token_seq, pointer_tensor, cell_text_data):
    """Decode otsl token seq from token seq and pointer prediction

    Args:
        otsl_token_seq List[str]: token sequence
        point_prediction torch.Tensor: pointer prediction
        cell_text_data List[str]: cell text data

    Returns:
        output_seq_tokens str: html sequence
    """
    # decode OTSL seq prediction output to html
    cell_text = None
    OTSL_full_compilation = []
    OTSL_row_compilation = []
    curr_column_index = 0

    for data_ind, token in enumerate(
        otsl_token_seq[2:]
    ):  # ignore the first two tokens as they are [<s> and <s_start>]
        if token == "C-tag":
            mapping_mask = pointer_tensor[data_ind]  # (bbox_token_cnt,)

            # mapping_mask is a boolean mask. Get all indices where value is True
            coord_indices = torch.nonzero(mapping_mask).squeeze(-1)  # (num_of_coords,)
            if len(coord_indices) == 0:  # No coordinate mapping predicted
                cell_text = None
            else:
                indices_list = coord_indices.tolist()
                for coord_ind in indices_list:
                    if coord_ind == 0:
                        continue
                    elif coord_ind > len(cell_text_data):
                        continue
                    else:
                        if cell_text is None:
                            cell_text = cell_text_data[coord_ind - 1]
                        else:
                            cell_text += " " + cell_text_data[coord_ind - 1]

            OTSL_row_compilation.append([1, 0, 0, cell_text])
            curr_column_index += 1
            cell_text = None
        elif token == "NL-tag":
            # new line
            OTSL_full_compilation.append(OTSL_row_compilation)
            OTSL_row_compilation = []
            curr_column_index = 0
        elif token == "L-tag":
            # column span
            for col_i in range(len(OTSL_row_compilation)):
                # traverse backwards
                col_i_value = OTSL_row_compilation[-1 - col_i]
                if col_i_value is not None:
                    col_i_value[2] += 1
                    break
            OTSL_row_compilation.append(None)
            curr_column_index += 1

        elif token == "U-tag":
            # row span
            for row_i in range(len(OTSL_full_compilation)):
                # traverse backwards
                row_i_value = OTSL_full_compilation[-1 - row_i]  # row_i_value is list
                if (
                    curr_column_index < len(row_i_value)
                    and row_i_value[curr_column_index] is not None
                ):
                    row_i_value[curr_column_index][1] += 1
                    break

            OTSL_row_compilation.append(None)
            curr_column_index += 1
        elif token == "X-tag":
            OTSL_row_compilation.append(None)
            curr_column_index += 1
            continue
        else:
            continue

    if len(OTSL_row_compilation) > 0:
        OTSL_full_compilation.append(OTSL_row_compilation)

    # unravel
    OTSL_full_compilation = [
        item for sublist in OTSL_full_compilation for item in sublist
    ]
    output_html_seq = ""
    current_data_index = 0
    for data_ind, token in enumerate(
        otsl_token_seq[2:]
    ):  # ignore the first two tokens as they are [<s> and <s_start>]
        if token in ["L-tag", "U-tag", "X-tag"]:
            current_data_index += 1
            continue
        elif token == "C-tag":
            cell_info = OTSL_full_compilation[current_data_index]
            if cell_info is not None:
                if cell_info[1] == 0 and cell_info[2] == 0:  # This is NOT a span cell
                    if cell_info[3] is None:
                        output_html_seq += "<td></td>"
                    else:
                        output_html_seq += "<td>" + cell_info[3] + "</td>"
                elif cell_info[1] == 0:  # This is column span
                    if cell_info[3] is None:
                        output_html_seq += '<td colspan="%s"></td>' % (cell_info[2] + 1)
                    else:
                        output_html_seq += (
                            '<td colspan="%s">' % (cell_info[2] + 1)
                            + cell_info[3]
                            + "</td>"
                        )
                elif cell_info[2] == 0:  # This is row span
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s"></td>' % (cell_info[1] + 1)
                    else:
                        output_html_seq += (
                            '<td rowspan="%s">' % (cell_info[1] + 1)
                            + cell_info[3]
                            + "</td>"
                        )
                else:  # This is both column and row span
                    if cell_info[3] is None:
                        output_html_seq += '<td rowspan="%s" colspan="%s"></td>' % (
                            cell_info[1] + 1,
                            cell_info[2] + 1,
                        )
                    else:
                        output_html_seq += (
                            '<td rowspan="%s" colspan="%s">'
                            % (cell_info[1] + 1, cell_info[2] + 1)
                            + cell_info[3]
                            + "</td>"
                        )

            current_data_index += 1

        elif token == "NL-tag":
            output_html_seq += "</tr><tr>"
        else:
            if token == "▁":
                token_to_add = " "
            else:
                token_to_add = token.replace("▁", "")
            output_html_seq += token_to_add

    # Formatting refinement
    if not output_html_seq.startswith("<thead>"):
        if "<thead>" in output_html_seq:
            output_html_seq = "<thead><tr>" + output_html_seq.split("<thead>", 1)[1]
        else:
            output_html_seq = "<thead><tr>" + output_html_seq
    else:
        output_html_seq = "<thead><tr>" + output_html_seq.split("<thead>", 1)[1]

    # Remove the last <tr> tag
    tmp_split = output_html_seq.rsplit("<tr>", 1)
    output_html_seq = tmp_split[0] + tmp_split[1]
    output_html_seq = output_html_seq.replace("<tr></thead>", "</thead>")
    output_html_seq = output_html_seq.replace("<tbody><td", "<tbody><tr><td")

    # Remove <pad> token
    output_html_seq = output_html_seq.replace("<pad>", "")

    return output_html_seq


def custom_format_html(html_string, tokenizer):
    """Custom format html string

    Args:
        html_string str: html string
    """
    tokens_to_remove = [
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        "<s_answer>",
        "</s_answer>",
    ]
    for token in tokens_to_remove:
        html_string = html_string.replace(token, "")

    html_seq = "<html><body><table>" + html_string + "</table></body></html>"

    return html_string, html_seq

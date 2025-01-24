import argparse
import datetime
import importlib
import os
from pathlib import Path

from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.plugins import CheckpointIO
import torch

from tflop.lightning_module.lightning_module import DataPLModule, TFLOPModelPLModule
from tflop.utils import (
    save_config_file,
    set_seed,
    set_up_logger_and_callbacks,
    set_up_tokenizer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        del checkpoint["state_dict"]
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {
            "model." + key: value for key, value in state_dict.items()
        }
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


def train(config):
    # Seed everything
    set_seed(config.get("seed", 42))

    # sanity check on Contrastive Learning setting
    assert any(
        [
            config.get("use_RowWise_contLearning", False),
            config.get("use_ColWise_contLearning", False),
        ]
    ), "Contrastive Learning setting is not correct."

    # Setup tokenizer
    tokenizer = set_up_tokenizer(
        pretrained_tokenizer_name_or_path=config.get(
            "pretrained_tokenizer_name_or_path", "hyunwoongko/asian-bart-ecjk"
        ),
        bbox_special_tokens=config.bbox_special_tokens,
        other_special_tokens=config.special_chars,
    )

    # Setup model PL module
    model_module = TFLOPModelPLModule(config=config, tokenizer=tokenizer, mode="train")
    # Setup data PL module
    data_module = DataPLModule(config=config)

    # Instantiate dataset
    dataset_collection = {}
    dataset_class = getattr(
        importlib.import_module(config.dataset_script_path), config.dataset_class_name
    )
    for split in ["train", "validation"]:
        dataset_collection[split] = dataset_class(
            tokenizer=tokenizer, split=split, config=config
        )
    data_module.train_dataset = dataset_collection["train"]
    data_module.val_dataset = dataset_collection["validation"]

    # Setup logger, callbacks and progressbar
    logger, lr_callback, checkpoint_callback, bar = set_up_logger_and_callbacks(config)

    custom_ckpt = CustomCheckpointIO()
    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices=4,
        strategy=config.get("strategy", "ddp"),
        accelerator="gpu",
        plugins=custom_ckpt,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        gradient_clip_val=config.gradient_clip_val,
        precision="bf16",
        num_sanity_val_steps=1,
        logger=logger,
        accumulate_grad_batches=config.get("accumulate_grad_batches", 1),
        callbacks=[lr_callback, checkpoint_callback, bar],
    )
    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume_from_checkpoint_path", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_config", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    exp_config = OmegaConf.load(args.exp_config)
    data_config = OmegaConf.load(args.data_config)
    cli_config = OmegaConf.from_cli(
        left_argv
    )  # config from cli in the form of key=value

    config = OmegaConf.unsafe_merge(exp_config, data_config, cli_config)
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not config.exp_version
        else config.exp_version
    )
    # Load bbox tokens into config
    bbox_special_tokens = [
        f"<bbox_{i}>" for i in range(max(config.input_size.values()) + 1)
    ]
    config.bbox_special_tokens = bbox_special_tokens

    OmegaConf.resolve(config)
    for sanity_config in ["result_path", "exp_name", "exp_version"]:
        assert (
            config.get(sanity_config, None) is not None
        ), f"{sanity_config} is not set in config."

    save_config_file(
        config, Path(config.result_path) / config.exp_name / config.exp_version
    )

    # print config on console
    for k, v in config.items():
        if k != "bbox_special_tokens":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {len(v)}")
    train(config)

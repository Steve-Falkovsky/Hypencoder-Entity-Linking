import os
from typing import Optional

import fire
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
    load_from_disk,
)
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from hypencoder_cb.modeling.hypencoder import (
    HypencoderDualEncoder,
    HypencoderDualEncoderConfig,
    TextDualEncoder,
)
from hypencoder_cb.modeling.shared import BaseDualEncoderConfig
from hypencoder_cb.train.args import (
    HypencoderDataConfig,
    HypencoderModelConfig,
    HypencoderTrainerConfig,
    HypencoderTrainingConfig,
)
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator

DEFAULT_CACHE_DIR = os.environ.get(
    "HYPENCODER_CACHE_DIR", os.path.expanduser("~/.cache/hypencoder")
)


def load_model(model_config: HypencoderModelConfig):
    config_cls_lookup = {
        "hypencoder": HypencoderDualEncoderConfig,
        "biencoder": BaseDualEncoderConfig,
    }

    model_cls_lookup = {
        "hypencoder": HypencoderDualEncoder,
        "biencoder": TextDualEncoder,
    }

    config_cls = config_cls_lookup[model_config.model_type]
    model_cls = model_cls_lookup[model_config.model_type]

    config = config_cls(
        query_encoder_kwargs=OmegaConf.to_container(
            model_config.query_encoder_kwargs
        ),
        passage_encoder_kwargs=OmegaConf.to_container(
            model_config.passage_encoder_kwargs
        ),
        loss_type=OmegaConf.to_container(model_config.loss_type),
        loss_kwargs=OmegaConf.to_container(model_config.loss_kwargs),
        shared_encoder=model_config.shared_encoder,
    )

    if model_config.checkpoint_path is not None:
        print("\n\n\n\nMODEL\n\n\n\n")
        # print(model_config.checkpoint_path)
        model = model_cls.from_pretrained(
            model_config.checkpoint_path, config=config
        )
        print(model)
    else:
        # print('AAAA\nAAAA\n\n\n\n')
        model = model_cls(config)

    return model


def load_data(data_config: HypencoderDataConfig):
    cache_dir = os.environ.get("HF_HOME", DEFAULT_CACHE_DIR)

    training_sources = [
        data_config.training_data_jsonl is not None,
        data_config.training_huggingface_dataset is not None,
        data_config.training_huggingface_disk_path is not None,
    ]
    if sum(training_sources) != 1:
        raise ValueError(
            "Must specify exactly one of training_data_jsonl, "
            "training_huggingface_dataset, or training_huggingface_disk_path"
        )

    validation_sources = [
        data_config.validation_data_jsonl is not None,
        data_config.validation_huggingface_dataset is not None,
        data_config.validation_huggingface_disk_path is not None,
    ]
    if sum(validation_sources) > 1:
        raise ValueError(
            "Must specify at most one of validation_data_jsonl, "
            "validation_huggingface_dataset, or validation_huggingface_disk_path"
        )

    if data_config.training_huggingface_dataset is not None:
        training_data = load_dataset(
            data_config.training_huggingface_dataset,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.training_huggingface_disk_path is not None:
        training_data = load_from_disk(data_config.training_huggingface_disk_path)
    else:
        training_data = load_dataset(
            "json",
            data_files=data_config.training_data_jsonl,
            split=data_config.training_data_split,
            cache_dir=cache_dir,
        )

    if isinstance(training_data, (DatasetDict, IterableDatasetDict)):
        training_data = training_data[data_config.training_data_split]

    validation_data = None
    if data_config.validation_huggingface_dataset is not None:
        validation_data = load_dataset(
            data_config.validation_huggingface_dataset,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )
    elif data_config.validation_huggingface_disk_path is not None:
        validation_data = load_from_disk(data_config.validation_huggingface_disk_path)
    elif data_config.validation_data_jsonl is not None:
        validation_data = load_dataset(
            "json",
            data_files=data_config.validation_data_jsonl,
            split=data_config.validation_data_split,
            cache_dir=cache_dir,
        )

    if isinstance(validation_data, (DatasetDict, IterableDatasetDict)):
        validation_data = validation_data[data_config.validation_data_split]

    return training_data, validation_data


def _normalize_training_arguments_kwargs(kwargs: dict) -> dict:
    """Normalize common legacy config keys to Hugging Face TrainingArguments."""
    kwargs = dict(kwargs)

    # Back-compat: older configs used `eval_strategy`.
    if "eval_strategy" in kwargs and "evaluation_strategy" not in kwargs:
        kwargs["evaluation_strategy"] = kwargs["eval_strategy"]
    kwargs.pop("eval_strategy", None)

    return kwargs


def get_collator(
    data_config: HypencoderDataConfig,
    trainer_config: HypencoderTrainerConfig,
    tokenizer,
):
    return GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=data_config.num_negatives_to_sample,
        positive_filter=data_config.positive_filter_type,
        positive_filter_kwargs=data_config.positive_filter_kwargs,
        positive_sampler="random",
        negative_sampler="random",
        num_positives_to_sample=data_config.num_positives_to_sample,
        label_key=data_config.label_key,
        query_padding_mode="longest",
    )


def load_tokenizer(model_config: HypencoderModelConfig):
    return AutoTokenizer.from_pretrained(
        model_config.tokenizer_pretrained_model_name_or_path
    )


def train_model(cfg: HypencoderTrainingConfig):
    print(cfg)
    resume_from_checkpoint = cfg.trainer_config.resume_from_checkpoint

    training_data, validation_data = load_data(cfg.data_config)
    tokenizer = load_tokenizer(cfg.model_config)
    
    model = load_model(cfg.model_config)
    print("Model loaded\n")
    
    collator = get_collator(cfg.data_config, cfg.trainer_config, tokenizer)
    print("data collated\n")
    
    train_arguments_kwargs = None
    hf_trainer_config = cfg.trainer_config.hf_trainer_config

    if OmegaConf.is_config(hf_trainer_config):
        train_arguments_kwargs = OmegaConf.to_container(hf_trainer_config)
    else:
        train_arguments_kwargs = hf_trainer_config.__dict__

    if train_arguments_kwargs is None:
        raise ValueError("hf_trainer_config resolved to None")
    if not isinstance(train_arguments_kwargs, dict):
        raise TypeError(
            f"hf_trainer_config must resolve to a dict, got {type(train_arguments_kwargs)}"
        )

    train_arguments_kwargs = _normalize_training_arguments_kwargs(train_arguments_kwargs)

    training_args = TrainingArguments(
        **train_arguments_kwargs,
    )
    
    print("training arguments loaded\n")
    
    print(validation_data)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=collator,
    )

    print("Starting training")
    if resume_from_checkpoint is True:
        output_dir = training_args.output_dir
        if output_dir is None:
            resume_from_checkpoint = False
        elif not os.path.exists(output_dir) or not any(
            [
                "checkpoint" in name
                for name in os.listdir(output_dir)
            ]
        ):
            resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def run_training(config_path: Optional[str] = None) -> None:
    schema = OmegaConf.structured(HypencoderTrainingConfig)

    if config_path is not None:
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(schema, config)
    else:
        config = schema

    train_model(config)


if __name__ == "__main__":
    fire.Fire(run_training)

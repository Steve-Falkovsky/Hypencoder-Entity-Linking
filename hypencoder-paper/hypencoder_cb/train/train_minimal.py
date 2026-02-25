import os
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments

from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder, HypencoderDualEncoderConfig
from hypencoder_cb.train.args import HypencoderTrainingConfig
from hypencoder_cb.train.data_collator import GeneralDualEncoderCollator


def main(config_path: str):
    schema = OmegaConf.structured(HypencoderTrainingConfig)
    cfg = OmegaConf.merge(schema, OmegaConf.load(config_path))

    mc = cfg.model_config
    dc = cfg.data_config
    tc = cfg.trainer_config.hf_trainer_config

    model_cfg = HypencoderDualEncoderConfig(
        query_encoder_kwargs=OmegaConf.to_container(mc.query_encoder_kwargs),
        passage_encoder_kwargs=OmegaConf.to_container(mc.passage_encoder_kwargs),
        loss_type=OmegaConf.to_container(mc.loss_type),
        loss_kwargs=OmegaConf.to_container(mc.loss_kwargs),
        shared_encoder=mc.shared_encoder,
    )

    if mc.checkpoint_path:
        model = HypencoderDualEncoder.from_pretrained(mc.checkpoint_path, config=model_cfg)
    else:
        model = HypencoderDualEncoder(model_cfg)

    tokenizer = AutoTokenizer.from_pretrained(mc.tokenizer_pretrained_model_name_or_path)

    train_ds = load_dataset(
        "json" if dc.training_data_jsonl else dc.training_huggingface_dataset,
        data_files=dc.training_data_jsonl if dc.training_data_jsonl else None,
        split=dc.training_data_split,
    )

    eval_ds = None
    if dc.validation_data_jsonl or dc.validation_huggingface_dataset:
        eval_ds = load_dataset(
            "json" if dc.validation_data_jsonl else dc.validation_huggingface_dataset,
            data_files=dc.validation_data_jsonl if dc.validation_data_jsonl else None,
            split=dc.validation_data_split,
        )

    collator = GeneralDualEncoderCollator(
        tokenizer=tokenizer,
        num_negatives_to_sample=dc.num_negatives_to_sample,
        positive_filter=dc.positive_filter_type,
        positive_filter_kwargs=dc.positive_filter_kwargs,
        positive_sampler="random",
        negative_sampler="random",
        num_positives_to_sample=dc.num_positives_to_sample,
        label_key=dc.label_key,
        query_padding_mode="longest",
    )

    args = TrainingArguments(**OmegaConf.to_container(tc))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=cfg.trainer_config.resume_from_checkpoint)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
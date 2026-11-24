import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional,Literal

import numpy as np
import torch
# import torch_npu ## Uncomment this line if you are using Ascend NPU

from datasets import Dataset, DatasetDict
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorWithPadding,
    set_seed
    ## TODO: import other classes you need
)
import evaluate
import peft
import adapters
import wandb
from peft import LoraConfig,TaskType,get_peft_model

from .dataHelper import get_dataset

os.environ["WANDB_MODE"] = "offline"

@dataclass
class BaseArgs:
    '''Base arguments for the training script.'''

    ## TODO: define your arguments here and set default values or fields
    dataset: str=None
    model_name: str=None
    sep_token: str='<SEP>'
    peft: Optional[Literal["adapter", "lora"]] = None
    max_length: int=256

@dataclass
class AdapterArgs:
    reduction_factor: int = 16
    non_linearity: str = "relu"
    mh_adapter: bool = False
    output_adapter: bool = True
    adapter_dropout: float = 0.1

@dataclass
class LoraArgs:
    '''Arguments for LoRA.'''

    ## TODO: define your arguments here and set default values or fields
    rank: int = 16
    alpha: int = 32
    lora_dropout: float = 0.1
    task_type = TaskType.SEQ_CLS


logger = logging.getLogger(__name__)


def print_trainable_parameters(model):
    """
    Print out the number of trainable parameters in the model.
    transformers models have a `print_trainable_parameters` method, but not all models have it.
    """

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {all_params:,} || "
          f"trainable%: {100 * trainable_params / all_params:.2f}%")


def parse_arguments():
    parser = HfArgumentParser((BaseArgs, TrainingArguments, AdapterArgs, LoraArgs))
    return parser.parse_args_into_dataclasses()


def set_random_seed(seed: int) -> None:
    set_seed(seed)


def set_logger() -> None:
    '''Set up the logger to print messages to stdout.'''

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout, 
        force=True,  
    )

    for name in ("urllib3", "filelock", "httpx", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_data(dataset_name: str,
              sep_token: str = "<SEP>") -> Tuple[DatasetDict, int]:
    '''Load dataset and return the number of labels.'''

    ## TODO: load dataset using `get_dataset()`
    raw_dataset = get_dataset(dataset_name,sep_token)
    labels = max(max(raw_dataset['train']['labels']),max(raw_dataset['test']['labels']))
    num_labels=labels+1
    return raw_dataset, num_labels


def get_model(model_name: str, num_labels: int):
    '''Load model and tokenizer.'''

    config = AutoConfig.from_pretrained(
        model_name
    )
    config.num_labels=num_labels
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model_options = {}
    if model_name == "facebook/bart-large":
        # This Hub revision adds safetensors for the original BART weights.
        model_options = {
            "revision": "4422181a85458e2acc57a4c1f845b9dfd60bd411",
            "use_safetensors": True,
        }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=config, **model_options
    )
    model.config.pad_token_id=tokenizer.pad_token_id

    ## TODO: set pad_token for tokenizer and model (Why?)

    return tokenizer, model


def tokenize_data(raw_dataset, tokenizer, max_length: int = 256):
    def preprocess(example):
        return tokenizer(
            example['text'],
            truncation=True,
            max_length=max_length
        )
    train_dataset, eval_dataset = raw_dataset['train'].map(preprocess,batched=True), raw_dataset['test'].map(preprocess,batched=True)
    return train_dataset, eval_dataset


def get_lora_model(model, lora_args):
    '''Initialize the model with LoRA modules.'''

    lora_config = LoraConfig(
        r=lora_args.rank,
        lora_alpha=lora_args.alpha,
        lora_dropout=lora_args.lora_dropout,
        task_type=lora_args.task_type,
        modules_to_save=["classification_head"] if model.config.model_type == "bart" else None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_adapter_model(model, adapter_args):
    '''Initialize the model with Bottleneck Adapter modules.'''

    adapters.init(model)
    adapter_config = adapters.SeqBnConfig(
        reduction_factor=adapter_args.reduction_factor,
        non_linearity=adapter_args.non_linearity,
        mh_adapter=adapter_args.mh_adapter,
        output_adapter=adapter_args.output_adapter,
        dropout=adapter_args.adapter_dropout,
    )
    model.add_adapter("bottleneck", config=adapter_config, set_active=True)
    model.train_adapter("bottleneck")
    print_trainable_parameters(model)
    return model


def get_data_collator(tokenizer):
    '''Define data collator for padding.'''

    ## TODO: define data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )
    return data_collator


def compute_metrics(pred) -> Dict[str, float]:
    '''Compute accuracy, macro F1 and micro F1.'''
    accuracy_metric=evaluate.load('accuracy')
    f1_metric=evaluate.load('f1')
    logits=pred.predictions
    labels=pred.label_ids
    predictions=np.argmax(logits,axis=-1)
    accuracy=accuracy_metric.compute(
        predictions=predictions,
        references=labels
    )['accuracy']
    macro_f1=f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='macro'
    )['f1']
    micro_f1=f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='micro'
    )['f1']
    weighted_f1=f1_metric.compute(
        predictions=predictions,
        references=labels,
        average='weighted'
    )['f1']
    ## TODO: compute metrics with `evaluate` package
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weight_f1": weighted_f1
    }


def preprocess_logits_for_metrics(logits, _labels):
    # BART returns its encoder hidden states alongside the classification logits.
    return logits[0] if isinstance(logits, tuple) else logits


def get_trainer(model, train_args, train_dataset, eval_dataset,
                tokenizer, data_collator):
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        processing_class=tokenizer,
    )
    return trainer


def main():
    # Parse arguments
    base_args, train_args, adapter_args, lora_args = parse_arguments()

    # Set seed before initializing model.
    set_random_seed(train_args.seed)

    # Set up logging
    set_logger()

    # load dataset
    raw_dataset, num_labels = load_data(base_args.dataset, base_args.sep_token)

    # get model and tokenizer
    tokenizer, model = get_model(base_args.model_name, num_labels)

    # tokenize dataset
    train_dataset, eval_dataset = tokenize_data(raw_dataset, tokenizer,
                                                base_args.max_length)

    # peft method
    if base_args.peft != None:
        if base_args.peft.lower() == "lora":
            model = get_lora_model(model, lora_args)
        elif base_args.peft.lower() == "adapter":
            model = get_adapter_model(model, adapter_args)
        else:
            raise ValueError("Unsupported PEFT method!")

    data_collator = get_data_collator(tokenizer)

    if train_args.process_index == 0:
        model_name = base_args.model_name.rsplit("/", 1)[-1]
        method = base_args.peft or "full"
        wandb.init(
            project="hw2-huggingface",
            name=train_args.run_name or f"{model_name}-{base_args.dataset}-{method}",
            config={
                "epoch": train_args.num_train_epochs,
                "batch_size": train_args.per_device_train_batch_size,
                "lr": train_args.learning_rate,
            },
        )

    trainer = get_trainer(model, train_args, train_dataset, eval_dataset,
                          tokenizer, data_collator)

    if train_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model(train_args.output_dir)
        trainer.save_state()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics, combined=True)

    if train_args.do_eval:
        logger.info("*** Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics, combined=True)

    if train_args.do_predict:
        logger.info("*** Predict ***")
        prediction = trainer.predict(eval_dataset)
        preds = np.argmax(prediction.predictions, axis=-1)
        with open("predict_results.txt", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(f"{int(p)}\n")

    if train_args.process_index == 0:
        wandb.finish()


if __name__ == "__main__":
    main()
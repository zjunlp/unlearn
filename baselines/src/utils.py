from pathlib import Path
import os
import torch
from typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch.nn as nn
import json
import re
import yaml

def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss

# def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
#     reduction = "sum" if num_items_in_batch is not None else "mean"
#     loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
#     if reduction == "sum":
#         loss = loss / num_items_in_batch
#     return loss

# def get_batch_loss(logits, labels, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()

#     # Flatten the tokens
#     shift_logits = shift_logits.view(-1, shift_logits.size(-1))
#     shift_labels = shift_labels.view(-1)
#     # Enable model parallelism
#     shift_labels = shift_labels.to(shift_logits.device)
#     loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
#     return loss


def get_rootpath():
    return str(Path(__file__).parent.resolve())


def get_basename(file_path: str):
    return os.path.basename(os.path.normpath(file_path))


def read_text(file_path: str) -> str:
    import pandas as pd

    if Path(file_path).suffix != '.txt':
        raise ValueError

    with open(file_path, 'r') as f:
        text: str = f.read()
    return text


def read_json(fpath: str):
    fpath = str(fpath)
    with open(fpath, 'r') as f:
        return json.load(f)


def output_json(data, fpath: str):
    fpath = str(fpath)
    assert fpath.endswith('.json')
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f: json.dump(data, f)


def file_exists(dir: str) -> bool:
    return os.path.isdir(dir) and any(os.path.isfile(os.path.join(dir, f)) for f in os.listdir(dir))


def output_text(data, fpath: str):
    fpath = str(fpath)
    assert fpath.endswith('.txt')
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f: f.write(data)


def load_model(
    model_dir: str,
    quantization_config: any = None,
) -> AutoModelForCausalLM:
    assert model_dir is not None
    if os.path.exists(os.path.join(model_dir, 'adapter_config.json')):
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map='cuda'
        )
    return model


def load_tokenizer(
    tokenizer_dir: str,
    add_pad_token: bool = True,
    use_fast: bool = True
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=use_fast) 
    if add_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_dir: str,
    model_name: str | None = None,
    tokenizer_dir: str | None = None,
    add_pad_token: bool = True,
    quantization_config: any = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = load_model(
        model_dir, quantization_config,
    )
    tokenizer = (load_tokenizer(tokenizer_dir, add_pad_token)
                 if tokenizer_dir is not None
                 else None)
    return model, tokenizer


def estimate_steps_per_epoch(samples: int,
                             epochs: int,
                             *_,
                             per_device_batch_size: int | None = None,
                             batch_size: int | None = None):
    """Overestimates number of steps per epoch.
    """
    from torch.cuda import device_count
    from math import ceil

    if per_device_batch_size is None and batch_size is None:
        raise ValueError("Either per_device_batch_size or batch_size must be specified.")
    if batch_size is None:
        # per_device_batch_size is specified
        cnt = device_count()
        if cnt == 0:
            raise ValueError("Device not detected.")
        batch_size: int = device_count() * per_device_batch_size

    samples_per_epoch = ceil(samples / epochs)
    steps_per_epoch = ceil(samples_per_epoch / batch_size)
    return steps_per_epoch


def pad_or_trim_tensor(tensor, target_length, padding_value=0):
    current_length = tensor.size(0)
    
    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    
    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor
    
    else:
        # No change needed
        return tensor

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("../config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_json(fpath: str):
    # load json or jsonl file
    fpath = str(fpath)
    try:
        with open(fpath, 'r') as f:
            data = json.load(f)
    except:
        with open(fpath, 'r') as f:
            data = [json.loads(line) for line in f]
    return data


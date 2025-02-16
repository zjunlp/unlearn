import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, Trainer

import wandb
import transformers
import os
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf
from src.utils import get_model_identifiers_from_yaml, find_all_linear_names
from src.dataset import QADataset, DefaultDataset

os.environ['WANDB_MODE'] = 'dryrun'

def finetune(cfg):
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    data_file = cfg.data_path
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    wandb.init(project='finetune', config={
    "learning_rate": cfg.lr,
    "epochs": cfg.num_epochs,
    "batch_size": batch_size * gradient_accumulation_steps * num_devices,
}, name=f'finetune-lr{cfg.lr}-epoch{cfg.num_epochs}')

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = cfg.max_length
    # torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)

    if "tofu" in data_file.lower() or "knowundo" in data_file.lower():
        print("using qa dataset..")
        dataset = QADataset(
            data_file,
            tokenizer=tokenizer,
            max_len=max_length
        )
    else:
        dataset = DefaultDataset(
            data_file,
            tokenizer=tokenizer,
            max_len=max_length
        )
    
    max_steps = int(cfg.num_epochs*len(dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_steps=max(1, max_steps//10),
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed=cfg.ds_config,
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r, 
            lora_alpha=cfg.LoRA.alpha, 
            target_modules=find_all_linear_names(model), 
            lora_dropout=cfg.LoRA.dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    #save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)


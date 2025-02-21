import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, Trainer
import wandb
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from src.utils import get_model_identifiers_from_yaml, find_all_linear_names, load_json, get_batch_loss
from src.dataset import ForgetRetainDataset, IDK_DPODataset,DPODataset ,choose_dataset
from omegaconf import OmegaConf
from iterative_trainer import IterativeUnlearner
from sure_trainer import SURE
from memflex_trainer import memflex

def unlearn(cfg):
    loss_type = cfg.loss_type
    retain_data_file = cfg.retain_data_path
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."
    
    forget_data_file = cfg.forget_data_path
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")
    project_name = getattr(cfg, 'project_name', 'my-unlearning-project')

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = cfg.max_length
    # if cfg.forget_loss == "dpo":
    #     torch_format_dataset = TextForgetDatasetDPOQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split)
    # else:
    #     torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss)

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)

    # Load reference model for specific loss types
    ref_model = (
        AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
        if 'npo' in loss_type or 'kl' in loss_type or 'dpo' in loss_type
        else None
    )

    if loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
        dpo_dataset = load_json(forget_data_file)
    else:
        # Instantiate the forget and retain datasets
        forget_dataset = choose_dataset(forget_data_file, tokenizer, max_len=max_length, model_cfg=model_cfg)
    retain_dataset = (choose_dataset(retain_data_file, tokenizer, max_len=max_length, model_cfg=model_cfg) if retain_data_file else None)

    # Create the combined dataset
    if loss_type in ["dpo","dpo_gdr","dpo_klr"]:
        dataset = IDK_DPODataset(
            forget_dataset=forget_dataset,
            idonknow_file_path=cfg.idonknow_file_path,
            retain_dataset=retain_dataset,
        )
    elif loss_type in "relearn":
        dataset = ForgetRetainDataset(
            forget_dataset=forget_dataset,
            retain_dataset=None,
        )
    elif loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
        dataset = DPODataset(
            data=dpo_dataset,
            tokenizer=tokenizer,
            max_len=max_length,
            retain_dataset=retain_dataset
        )
    else:
        dataset = ForgetRetainDataset(
            forget_dataset=forget_dataset,
            retain_dataset=retain_dataset,
        )

    steps_per_epoch = len(dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA.r, 
        lora_alpha=cfg.LoRA.alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA.dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA.r != 0:
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed=cfg.ds_config,
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            evaluation_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed,
            report_to="none",
        )
    
    if "sure" in cfg.loss_type:
        trainer = SURE(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset = dataset,
            compute_metrics=None,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type = loss_type,
        )
    elif "memflex" in cfg.loss_type:
        trainer = memflex(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset = dataset,
            compute_metrics=None,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type = loss_type,
        )
    else:
        trainer = IterativeUnlearner(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset = dataset,
            compute_metrics=None,
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type = loss_type,
        )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    # save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)
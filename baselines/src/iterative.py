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
from src.dataset import QADataset, TextDataset, ForgetRetainDataset, IDK_DPODataset,DPODataset ,choose_dataset
from omegaconf import OmegaConf
import copy
import deepspeed
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

os.environ['WANDB_MODE'] = 'dryrun'

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

    # Initialize wandb
    wandb.init(project=project_name, config={
        "learning_rate": cfg.lr,
        "epochs": cfg.num_epochs,
        "batch_size": batch_size * gradient_accumulation_steps,
    }, name=loss_type)
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
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
    else:
        trainer = IterativeUnlearner(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset = dataset,
            compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
            # callbacks=[GlobalStepDeletionCallback],
            args=training_args,
            data_collator=dataset.get_collate_fn(),
            loss_type = loss_type,
        )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 **kwargs):
        self.loss_type = kwargs.pop("loss_type", "ga")
        self.ref_model = kwargs.pop("ref_model", None)
        self.beta = kwargs.pop("beta", 0.1)    # Only relevant when `'po' in self.loss_type`

        super().__init__(*args, **kwargs)
        if self.ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            # ref_model = ref_model.eval()
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)



    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def compute_loss(self, model, x, return_outputs=False, num_items_in_batch=None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        ### 1. Split the input ###
        
        if self.loss_type in ["dpo","dpo_gdr","dpo_klr"]:
            x_f, x_r, x_i = x
        elif self.loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
            x_p, x_n, x_r = x
        else:
            x_f, x_r = x

        ### 2. Calculate Loss Based on Loss Type ###
        if self.loss_type == 'ga':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss
            loss = -loss_f

        elif self.loss_type == 'ga_gdr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            loss = -loss_f + loss_r

        elif self.loss_type == 'ga_klr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            loss = -loss_f + kl_r

        elif self.loss_type == 'npo':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        elif self.loss_type == 'npo_gdr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_npo = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_npo + loss_r

        elif self.loss_type == 'npo_klr':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_npo= -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_npo + kl_r

        elif self.loss_type in "relearn":
            assert x_r is None, "retain data is not None but loss type is relearn(gd)."     
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss = outputs_f.loss
            
        elif self.loss_type in ["relearn_klr", "relearn_klr_gdr", "relearn_gdr"]:
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss
            
            if self.loss_type == "relearn_gdr":
                loss = loss_f + loss_r
            elif self.loss_type in ["relearn_klr", "relearn_klr_gdr"]:
                with torch.no_grad():
                    outputs_r_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                    )
                
                outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
                outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])

                kl_r = F.kl_div(
                    outputs_r_logits,
                    outputs_r_ref_logits,
                    reduction='batchmean',
                    log_target=True
                )

                if self.loss_type == "relearn_klr":
                    loss = loss_f + kl_r
                elif self.loss_type == "relearn_klr_gdr":
                    loss = loss_f + kl_r + loss_r
                else:
                    raise NotImplementedError("Cannot infer the given loss type.")
        elif self.loss_type in ["relearn_dpo", "relearn_dpo_gdr", "relearn_dpo_klr"]:
            iwant_outputs = model(
                x_p['input_ids'],
                labels=x_p['labels'] if 'labels' in x_p else x_p['input_ids'].clone(),
                attention_mask=x_p['attention_mask'] if 'attention_mask' in x_p else torch.ones_like(x_p['input_ids'], dtype=torch.bool)
            )
            idontwant_outputs = model(
                x_n['input_ids'],
                labels=x_n['labels'] if 'labels' in x_n else x_n['input_ids'].clone(),
                attention_mask=x_n['attention_mask'] if 'attention_mask' in x_n else torch.ones_like(x_n['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                iwant_outputs_ref = self.ref_model(
                    x_p['input_ids'],
                    labels=x_p['labels'] if 'labels' in x_p else x_p['input_ids'].clone(),
                    attention_mask=x_p['attention_mask'] if 'attention_mask' in x_p else torch.ones_like(x_p['input_ids'], dtype=torch.bool)
                )
                idontwant_outputs_ref = self.ref_model(
                    x_n['input_ids'],
                    labels=x_n['labels'] if 'labels' in x_n else x_n['input_ids'].clone(),
                    attention_mask=x_n['attention_mask'] if 'attention_mask' in x_n else torch.ones_like(x_n['input_ids'], dtype=torch.bool)
                )
                iwant_loss_ref = -1 * iwant_outputs_ref.loss
                idontwant_loss_ref = -1 * idontwant_outputs_ref.loss
            
            iwant_loss = -1 * iwant_outputs.loss
            idontwant_loss = -1 * idontwant_outputs.loss

            pi_logratios = iwant_loss - idontwant_loss
            pi_logratios_ref = iwant_loss_ref - idontwant_loss_ref
            loss = -F.logsigmoid(self.beta * (pi_logratios - pi_logratios_ref)).mean() * 2 / self.beta

            if self.loss_type == "relearn_dpo_gdr":
                retain_outputs = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                loss = loss + retain_outputs.loss
            elif self.loss_type == "relearn_dpo_klr":
                with torch.no_grad():
                    retain_outputs_ref = self.ref_model(
                        x_r['input_ids'],
                        labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                        attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                    )
                retain_probs_ref = F.softmax(retain_outputs_ref.logits, dim=-1).view(-1, retain_outputs_ref.logits.shape[-1])

                retain_outputs = model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )
                retain_probs = F.softmax(retain_outputs.logits, dim=-1).view(-1, retain_outputs.logits.shape[-1])

                retain_loss = F.kl_div(
                    retain_probs,
                    retain_probs_ref,
                    reduction='batchmean',
                    log_target=True
                )

                loss = loss + retain_loss

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        ### 4. Log the Loss ###
        wandb.log({"loss": loss.item()})

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


# saliency map with neuron level mask
class SURE(Trainer):
    """Custom Trainer for Unlearning with Neuron-Level Saliency Map"""

    def __init__(self, *args,
                 loss_type: str = 'ga',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 alpha: float = 1.0,  # Weighting for retain data loss
                 threshold: int = 99,
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta    # Only relevant when 'npo' in self.loss_type
        self.alpha = alpha  # Weighting for retain data loss
        self.threshold = threshold

        super().__init__(*args, **kwargs)
        if self.ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            # ref_model = ref_model.eval()
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)

        print(f'Weight for utility constraint: {self.alpha}, Threshold to filter salient modules: {self.threshold}')

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model

    def compute_loss(self, model, x, return_outputs=False, num_items_in_batch=None):
        x_f, x_r = x

        # Reset saliency mask
        self.m_S = None

        ### Compute loss on forget data ###
        if self.loss_type == 'ga_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss
            loss = -loss_f
        elif self.loss_type == 'ga_gdr_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            loss = -loss_f + loss_r
        elif self.loss_type == 'ga_klr_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            loss_f = outputs_f.loss

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            loss = -loss_f + kl_r
        elif self.loss_type == 'npo_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            loss = loss_f
        elif self.loss_type == 'npo_gdr_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_f + loss_r
        elif self.loss_type == 'npo_klr_sure':
            outputs_f = model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
            )
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            kl_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_f= -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta 
            loss = loss_f + kl_r
        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        # Zero existing gradients
        self.optimizer.zero_grad()

        loss_f.backward(retain_graph=True)
        # Compute neuron-wise gradient norms within no_grad context
        with torch.no_grad():
            neuron_grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad = param.grad.detach().data.float()  # Cast to float32
                    if grad.dim() > 1:
                        # Compute the gradient norm per neuron along the first dimension
                        grad_norms_per_neuron = grad.norm(2, dim=list(range(1, grad.dim()))).cpu().numpy()
                    else:
                        # For 1D parameters (e.g., biases)
                        grad_norms_per_neuron = grad.abs().cpu().numpy()

                    for idx, grad_norm in enumerate(grad_norms_per_neuron):
                        neuron_name = f"{name}.{idx}"
                        neuron_grad_norms[neuron_name] = grad_norm

            # Determine threshold gamma (e.g., 90th percentile of gradient norms)
            grad_norms = list(neuron_grad_norms.values())
            gamma = np.percentile(grad_norms, self.threshold)

            # Create saliency mask at neuron level
            self.m_S = {neuron_name: 1.0 if norm >= gamma else 0.0 for neuron_name, norm in neuron_grad_norms.items()}
        
        return (loss, outputs_f) if return_outputs else loss

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
        ) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.

            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                torch.cuda.empty_cache()

            kwargs = {}

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
            
            # Apply neuron-wise mask to gradients if m_S is defined
            if hasattr(self, 'm_S') and self.m_S is not None:
                for name, param in model.named_parameters():
                    if 'lora' in name and param.grad is not None:
                        grad = param.grad
                        if grad.dim() > 1:
                            # Build the mask tensor per neuron
                            neuron_mask_values = [self.m_S.get(f"{name}.{idx}", 0.0) for idx in range(grad.shape[0])]
                            mask_shape = [grad.shape[0]] + [1]*(grad.dim()-1)
                            mask = torch.tensor(neuron_mask_values, device=grad.device, dtype=grad.dtype).view(*mask_shape)
                            grad.mul_(mask)
                        else:
                            # For 1D parameters (e.g., biases)
                            neuron_mask_values = [self.m_S.get(f"{name}.{idx}", 0.0) for idx in range(grad.shape[0])]
                            mask = torch.tensor(neuron_mask_values, device=grad.device, dtype=grad.dtype)
                            grad.mul_(mask)

            if num_items_in_batch is None:
                return loss.detach() / self.args.gradient_accumulation_steps
            return loss.detach()
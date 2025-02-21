import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import get_batch_loss
import copy
import deepspeed
import numpy as np
from typing import Any, Dict, Union

class memflex(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("loss_type", "ga")
        self.ref_model = kwargs.pop("ref_model", None)
        self.beta = kwargs.pop("beta", 0.1)    # Only relevant when `'po' in self.loss_type`
        # memflex特有的阈值
        self.threshold_f = kwargs.pop('threshold_f')
        self.threshold_r = kwargs.pop('threshold_r')
        self.sim_thresh = kwargs.pop('sim_thresh')
        self.grad_thresh = kwargs.pop('grad_thresh')
        self.ga_ratio = kwargs.pop('ga_ratio')
        self.gd_ratio = kwargs.pop('gd_ratio')
        self.count = 0

        super().__init__(*args, **kwargs)
        if self.ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
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
        if self.loss_type in ["dpo_gdr_memflex", "dpo_klr_memflex"]:
            x_f, x_r, x_i = x
        else:
            x_f, x_r = x

        ### 2. Calculate Loss Based on Loss Type ###
        if self.loss_type == 'ga_gdr_memflex':
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

            loss = -1 * self.ga_ratio * loss_f + self.gd_ratio * loss_r

        elif self.loss_type == 'ga_klr_memflex':
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

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            loss_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            loss = -1 * self.ga_ratio * loss_f + self.gd_ratio * loss_r

        elif self.loss_type == 'npo_gdr_memflex':
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
            loss = self.ga_ratio * loss_f + self.gd_ratio * loss_r

        elif self.loss_type == 'npo_klr_memflex':
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

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

            outputs_r_logits = F.log_softmax(outputs_r.logits, dim=-1).view(-1, outputs_r.logits.shape[-1])
            outputs_r_ref_logits = F.log_softmax(outputs_r_ref.logits, dim=-1).view(-1, outputs_r_ref.logits.shape[-1])
            loss_r = F.kl_div(
                outputs_r_logits,
                outputs_r_ref_logits,
                reduction='batchmean',
                log_target=True
            )

            outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
            outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
            neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
            loss_f = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            loss = self.ga_ratio * loss_f + self.gd_ratio * loss_r

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        # Zero existing gradients
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        grad_forget = {}
        grad_retain = {}

        for name, param in model.named_parameters():
            if 'lora' in name:
                grad_forget[name] = torch.zeros_like(param, device='cpu').float()
                grad_retain[name] = torch.zeros_like(param, device='cpu').float()

        # Calculate grad_forget
        loss_f.backward(retain_graph=True)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora' in name:
                    grad_forget[name] += param.grad.detach().cpu().float()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Calculate grad_retain
        loss_r.backward(retain_graph=True)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'lora' in name:
                    grad_retain[name] += param.grad.detach().cpu().float()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Localization
        delta_matrix = {}
        forget_list = []
        retain_list = []
        item_list = []
        
        for k, _ in grad_forget.items():
            if k in grad_retain:  # intersection of unlearn and retain
                delta_matrix[k] = compute_cosine_similarity(grad_forget[k], grad_retain[k]).squeeze()
                num_forget = np.mean(np.abs(grad_forget[k].numpy()))
                num_retain = np.mean(np.abs(grad_retain[k].numpy()))
                forget_list.append(num_forget)
                retain_list.append(num_retain)
                item_list.append(delta_matrix[k])

        sim_thre = self.sim_thresh
        grad_thre = self.grad_thresh
        item_array = np.array(item_list)
        forget_array = np.array(forget_list)
        forget_sim_idx = np.where(item_array < sim_thre)[0]
        forget_grad_idx = np.where(forget_array > grad_thre)[0]

        located_region_num = list(np.intersect1d(forget_sim_idx, forget_grad_idx))
        self.located_region = []
        for i, key in enumerate(grad_forget.keys()):
            if i in located_region_num:
                self.located_region.append(key)

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

        if hasattr(self, 'located_region') and self.located_region is not None:
            for name, param in self.model.named_parameters():
                if name not in self.located_region:
                    if param.grad is not None:
                        param.grad.zero_()

        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
        return loss.detach()

def compute_cosine_similarity(p, q):
    p = p.numpy()
    q = q.numpy()
    p = p.reshape(1, -1)
    q = q.reshape(1, -1)
    return cosine_similarity(p, q)
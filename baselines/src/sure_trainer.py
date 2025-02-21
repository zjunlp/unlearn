import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, Trainer
from src.utils import get_batch_loss
import copy
import deepspeed
import numpy as np
from typing import Any, Dict, Union

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
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer
from src.utils import get_batch_loss
import copy
import deepspeed

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

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

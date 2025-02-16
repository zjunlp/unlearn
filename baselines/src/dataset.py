from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple, Dict
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer

class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return; # end if Path(file_path).suffix == '.json'

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass    # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn

class BaseDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, max_len: int = 4096, add_bos_token: bool = True):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_len = max_len
        self.add_bos_token = add_bos_token
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.text_column = "text"
        self.label_column = "labels"

    def _pad_or_trim(self, input_sequence: torch.Tensor):
        return pad_or_trim_tensor(
            input_sequence,
            target_length=self.max_len,
            padding_value=self.tokenizer.pad_token_id
        )

    def __len__(self):
        return len(self.input_ids)

    def get_collate_fn(self):
        # Base collate function for subclasses to use or override.
        def collate_fn(batch: List[dict]):
            input_ids = torch.stack([b["input_ids"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])

            output = {
                "input_ids": input_ids,
                "labels": labels,
            }
            if "attention_mask" in batch[0]:
                attention_masks = torch.stack([b["attention_mask"] for b in batch])
                output["attention_mask"] = attention_masks

            return output

        return collate_fn


class TextDataset(BaseDataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 4096, add_bos_token: bool = True):
        super().__init__(tokenizer, max_len, add_bos_token)

        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0]:
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            for s in self.strings:
                encoding = tokenizer(s, add_special_tokens=add_bos_token, return_tensors='pt').input_ids[0]
                encoding = self._pad_or_trim(encoding)
                self.input_ids.append(encoding)

        elif Path(file_path).suffix == '.txt':
            tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
            if add_bos_token:
                self.input_ids = [
                    F.pad(tokens[i: i + max_len - 1], (1, 0), value=tokenizer.bos_token_id)
                    for i in range(0, len(tokens), max_len - 1)
                ]
            else:
                self.input_ids = [tokens[i: i + max_len] for i in range(0, len(tokens), max_len)]

            if len(self.input_ids[-1]) < max_len:
                self.input_ids[-1] = torch.concat([self.input_ids[-1], self.input_ids[0]], dim=-1)[:max_len]

        else:
            raise ValueError("Unsupported file type. Use '.json' or '.txt'.")

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.input_ids[index].clone()  # Labels are same as input_ids for causal language modeling
        }


class QADataset(BaseDataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 4096, add_bos_token: bool = True, model_cfg: dict = None):
        super().__init__(tokenizer, max_len, add_bos_token)

        if "tofu" in file_path.lower():
            self.text_column = "question"
            self.label_column = "answer"
            # tofu is jsonl format
            with open(file_path, 'r') as f:
                examples = [json.loads(line) for line in f]
        else:
            with open(file_path, 'r') as f:
                examples = json.load(f)
        if not isinstance(examples, list) or not all(isinstance(d, dict) and self.text_column in d and self.label_column in d for d in examples):
            raise ValueError("The JSON file must be a list of dictionaries with 'question' and 'answer' keys.")

        question_start_tag = model_cfg.get("question_start_tag", "")
        question_end_tag = model_cfg.get("question_end_tag", "")
        answer_tag = model_cfg.get("answer_tag", "")

        batch_size = len(examples) 
        max_length = max_len
        inputs = [question_start_tag+str(x[self.text_column])+question_end_tag for x in examples]
        targets = [answer_tag+str(x[self.label_column]) for x in examples]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets, add_special_tokens=False)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
            self.input_ids.append(model_inputs["input_ids"][i])
            self.attention_masks.append(model_inputs["attention_mask"][i])
            self.labels.append(labels["input_ids"][i])
        model_inputs["labels"] = labels["input_ids"]
        return

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index]
        }

class ForgetRetainDataset(Dataset):
    def __init__(self, forget_dataset: BaseDataset, retain_dataset: BaseDataset | None = None):
        self.forget_dataset = forget_dataset
        self.retain_exists = retain_dataset is not None
        self.retain_dataset = retain_dataset

    def __getitem__(self, index):
        forget_data = self.forget_dataset[index]
        if self.retain_exists:
            retain_data = self.retain_dataset[index % len(self.retain_dataset)]
            return forget_data, retain_data
        else:
            return forget_data, None

    def __len__(self):
        return len(self.forget_dataset)

    def get_collate_fn(self):
        def collate_fn(batch: List[Tuple[dict, dict]]):
            batch_forget = {
                "input_ids": torch.stack([pair[0]["input_ids"] for pair in batch]),
                "labels": torch.stack([pair[0]["labels"] for pair in batch]),
            }
            if "attention_mask" in batch[0][0]:
                batch_forget["attention_mask"] = torch.stack([pair[0]["attention_mask"] for pair in batch])

            if self.retain_exists:
                batch_retain = {
                    "input_ids": torch.stack([pair[1]["input_ids"] for pair in batch]),
                    "labels": torch.stack([pair[1]["labels"] for pair in batch]),
                }
                if "attention_mask" in batch[0][1]:
                    batch_retain["attention_mask"] = torch.stack([pair[1]["attention_mask"] for pair in batch])
            else:
                batch_retain = None

            return batch_forget, batch_retain

        return collate_fn

class DPODataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: AutoTokenizer, max_len: int = 1024, retain_dataset: BaseDataset | None = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.retain_exists = retain_dataset is not None
        self.retain_dataset = retain_dataset
        self.processed_data = []
        inputs = [str(x["question"]) for x in self.data]
        positive_targets = [str(x["positive_answer"]) for x in self.data]
        negative_targets = [str(x["negative_answer"]) for x in self.data]
        model_inputs = tokenizer(inputs)
        model_inputs["positive_input_ids"] = [None] * len(self.data)
        model_inputs["negative_input_ids"] = [None] * len(self.data)
        model_inputs["positive_attention_mask"] = [None] * len(self.data)
        model_inputs["negative_attention_mask"] = [None] * len(self.data)
        positive_labels = tokenizer(positive_targets, add_special_tokens=False)
        negative_labels = tokenizer(negative_targets, add_special_tokens=False)

        for i in range(len(self.data)):
            sample_input_ids = model_inputs["input_ids"][i]
            positive_label_ids = positive_labels["input_ids"][i] + [tokenizer.eos_token_id]
            negative_label_ids = negative_labels["input_ids"][i] + [tokenizer.eos_token_id]

            model_inputs["positive_input_ids"][i] = sample_input_ids + positive_label_ids
            model_inputs["negative_input_ids"][i] = sample_input_ids + negative_label_ids

            positive_labels["input_ids"][i] = [-100] * len(sample_input_ids) + positive_label_ids
            negative_labels["input_ids"][i] = [-100] * len(sample_input_ids) + negative_label_ids
            
            model_inputs["positive_attention_mask"][i] = [1] * len(model_inputs["positive_input_ids"][i])
            model_inputs["negative_attention_mask"][i] = [1] * len(model_inputs["negative_input_ids"][i])
        
        for i in range(len(self.data)):
            sample_positive_input_ids = model_inputs["positive_input_ids"][i]
            sample_negative_input_ids = model_inputs["negative_input_ids"][i]
            positive_label_ids = positive_labels["input_ids"][i]
            negative_label_ids = negative_labels["input_ids"][i]

            model_inputs["positive_input_ids"][i] = [tokenizer.pad_token_id] * (
                max_len - len(sample_positive_input_ids)
            ) + sample_positive_input_ids
            model_inputs["negative_input_ids"][i] = [tokenizer.pad_token_id] * (
                max_len - len(sample_negative_input_ids)
            ) + sample_negative_input_ids

            model_inputs["positive_attention_mask"][i] = [0] * (max_len - len(sample_positive_input_ids)) + model_inputs[
                "positive_attention_mask"
            ][i]
            model_inputs["negative_attention_mask"][i] = [0] * (max_len - len(sample_negative_input_ids)) + model_inputs[
                "negative_attention_mask"
            ][i]

            positive_labels["input_ids"][i] = [-100] * (max_len - len(sample_positive_input_ids)) + positive_label_ids
            negative_labels["input_ids"][i] = [-100] * (max_len - len(sample_negative_input_ids)) + negative_label_ids

            model_inputs["positive_input_ids"][i] = torch.tensor(model_inputs["positive_input_ids"][i][:max_len])
            model_inputs["negative_input_ids"][i] = torch.tensor(model_inputs["negative_input_ids"][i][:max_len])

            model_inputs["positive_attention_mask"][i] = torch.tensor(model_inputs["positive_attention_mask"][i][:max_len])
            model_inputs["negative_attention_mask"][i] = torch.tensor(model_inputs["negative_attention_mask"][i][:max_len])

            positive_labels["input_ids"][i] = torch.tensor(positive_labels["input_ids"][i][:max_len])
            negative_labels["input_ids"][i] = torch.tensor(negative_labels["input_ids"][i][:max_len])

            self.processed_data.append({
                "positive_input_ids": model_inputs["positive_input_ids"][i],
                "positive_attention_mask": model_inputs["positive_attention_mask"][i],
                "positive_labels": positive_labels["input_ids"][i],
                "negative_input_ids": model_inputs["negative_input_ids"][i],
                "negative_attention_mask": model_inputs["negative_attention_mask"][i],
                "negative_labels": negative_labels["input_ids"][i],
            })

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        dpo_data = self.processed_data[index]
        if self.retain_exists:
            retain_data = self.retain_dataset[index % len(self.retain_dataset)]
            return dpo_data, retain_data
        else:
            return dpo_data, None

    def get_collate_fn(self):
        def collate_fn(batch: List[Tuple[dict, dict]]):
            batch_positive = {
                "input_ids": torch.stack([pair[0]["positive_input_ids"] for pair in batch]),
                "labels": torch.stack([pair[0]["positive_labels"] for pair in batch]),
            }
            if "positive_attention_mask" in batch[0][0]:
                batch_positive["attention_mask"] = torch.stack([pair[0]["positive_attention_mask"] for pair in batch])

            batch_negative = {
                "input_ids": torch.stack([pair[0]["negative_input_ids"] for pair in batch]),
                "labels": torch.stack([pair[0]["negative_labels"] for pair in batch]),
            }
            if "negative_attention_mask" in batch[0][0]:
                batch_negative["attention_mask"] = torch.stack([pair[0]["negative_attention_mask"] for pair in batch])

            if self.retain_exists:
                batch_retain = {
                    "input_ids": torch.stack([pair[1]["input_ids"] for pair in batch]),
                    "labels": torch.stack([pair[1]["labels"] for pair in batch]),
                }
                if "attention_mask" in batch[0][1]:
                    batch_retain["attention_mask"] = torch.stack([pair[1]["attention_mask"] for pair in batch])
            else:
                batch_retain = None

            return batch_positive, batch_negative, batch_retain

        return collate_fn

class IDK_DPODataset(Dataset):
    def __init__(
        self,
        forget_dataset: BaseDataset,
        idonknow_file_path: str,
        retain_dataset: BaseDataset | None = None
    ):
        self.forget_dataset = forget_dataset
        
        with open(idonknow_file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()

        # Tokenize each line from the txt file
        self.idonknow_dataset = [
            forget_dataset.tokenizer(
                line.strip(), 
                add_special_tokens=True, 
                return_tensors='pt', 
                max_length=forget_dataset.max_len, 
                padding="max_length", 
                truncation=True
            ).input_ids[0]
            for line in data if line.strip()  # Skip empty lines
        ]

        self.retain_exists = retain_dataset is not None
        self.retain_dataset = retain_dataset

    def __getitem__(self, index):
        forget_data = self.forget_dataset[index]
        idonknow_data = self.idonknow_dataset[index % len(self.idonknow_dataset)]

        if self.retain_exists:
            retain_data = self.retain_dataset[index % len(self.retain_dataset)]
            return forget_data, retain_data, idonknow_data
        else:
            return forget_data, None, idonknow_data

    def __len__(self):
        return len(self.forget_dataset)

    def get_collate_fn(self):
        def collate_fn(batch: List[Tuple[dict, dict, torch.Tensor]]):
            # Split the batch into forget, retain, and idonknow components
            batch_forget = {
                "input_ids": torch.stack([pair[0]["input_ids"] for pair in batch]),
                "labels": torch.stack([pair[0]["labels"] for pair in batch]),
            }
            if "attention_mask" in batch[0][0]:
                batch_forget["attention_mask"] = torch.stack([pair[0]["attention_mask"] for pair in batch])

            batch_idonknow = {
                "input_ids": torch.stack([pair[2] for pair in batch]),
            }

            if self.retain_exists:
                batch_retain = {
                    "input_ids": torch.stack([pair[1]["input_ids"] for pair in batch]),
                    "labels": torch.stack([pair[1]["labels"] for pair in batch]),
                }
                if "attention_mask" in batch[0][1]:
                    batch_retain["attention_mask"] = torch.stack([pair[1]["attention_mask"] for pair in batch])
            else:
                batch_retain = None

            return batch_forget, batch_retain, batch_idonknow

        return collate_fn


# Choose dataset based on file type or structure
def choose_dataset(file_path: str, tokenizer, max_len: int, add_bos_token: bool = True, model_cfg: dict = None):
    text_column = "text"
    label_column = "labels"
    json_type = "json"
    if "tofu" in file_path.lower():
        text_column = "question"
        label_column = "answer"
        json_type = "jsonl"
    if Path(file_path).suffix == '.txt':
        return TextDataset(file_path, tokenizer, max_len=max_len, add_bos_token=add_bos_token)
    else:
        # Load data to determine dataset type
        with open(file_path, 'r') as f:
            if json_type == "json":
                data = json.load(f)
            elif json_type == "jsonl":
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported JSON type")
            
        if isinstance(data, list) and all(isinstance(d, dict) and text_column in d and label_column in d for d in data):
            return QADataset(file_path, tokenizer, max_len=max_len, add_bos_token=add_bos_token, model_cfg=model_cfg)
        else:
            return TextDataset(file_path, tokenizer, max_len=max_len, add_bos_token=add_bos_token)

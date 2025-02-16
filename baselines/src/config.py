import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True,
                                         llm_int8_threshold=200.0)

load_config = {
    "torch_dtype": torch.bfloat16,
    "low_cpu_mem_usage": True,
    "device_map": "auto",
    "quantization_config": quantization_config,
}

MAX_LEN_TOKENS = 4096   # Context length LLaMA 2

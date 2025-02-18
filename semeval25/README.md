# SemEval Unlearning
This folder contains the solution developed by ZJUKLAB for the [SemEval 2025 Task 4](https://llmunlearningsemeval2025.github.io/) competition.

## Installation


```bash
conda create -n semeval_unlearn python=3.12
conda activate semeval_unlearn
pip install -r requirements.txt
```

### Script Arguments

- `--forget_dataset`: Specifies the dataset to forget (must be a valid dataset path or identifier).
- `--retain_dataset`: Specifies the dataset to retain.
- `--model_path`: Path to the pre-trained model.
- `--output_dir`: Directory where results and logs will be saved.

### Run the Script:

```bash
torchrun --nproc_per_node=1 --master_port=29500 unlearn-merging.py --forget_dataset /path/to/forget_data --retain_dataset /path/to/retain_data --model_path /path/to/model --output_dir /path/to/output
```
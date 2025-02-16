import json
import os
import yaml
import argparse
import random

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format: {file_path}")
        return None

def find_matching_answer(data, query):
    for item in data:
        if item['query'] == query:
            return item['generated_response']
    return None


def generate_candidates(data_dir, model_config, output_prefix, candidate_type):
    """
    Prepare candidates for evaluation.

    Args:
        data_dir (str)
        model_config (dict)
        output_prefix (str)
        candidate_type (str)
    """
    
    pretrain_file = os.path.join(data_dir, f'{output_prefix}_pretrained__model__{candidate_type}.json')

    pretrain_data = load_json(pretrain_file)
    if not pretrain_data:
        return []
    
    random.seed(42)
    if "tofu" in output_prefix.lower():
        pretrain_data = random.sample(pretrain_data, 200)

    # load ckpt responses
    model_responses = {}
    for method, config in model_config.items():
        key = config["answer_key"]
        response = load_json(os.path.join(data_dir, config[candidate_type]))
        model_responses[key] = response
    
    candidates = []
    for idx, pretrain_item in enumerate(pretrain_data):
        candidate_item = {}
        candidate_item['id'] = f'{candidate_type}_{idx}'
        candidate_item['question'] = pretrain_item['query']
        candidate_item['model_answer0'] = pretrain_item['generated_response']
    
        for model_answer_key, response in model_responses.items():
            if response is None:
                breakpoint()
            answer = find_matching_answer(response, pretrain_item['query'])
            if answer:
                candidate_item[model_answer_key] = answer
        candidates.append(candidate_item)

    output_file = os.path.join(data_dir, f'{output_prefix}_{candidate_type}_candidates.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(candidates)} {candidate_type} candidates to {output_file}")
    
    return candidates

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../kud-llama-results')
    parser.add_argument('--config_path', type=str, default='./config/datapre.yaml')
    parser.add_argument('--output_prefix', type=str, default='llama2-7b_kud')
    args = parser.parse_args()

    config = load_config(args.config_path)
    if not config:
        exit()

    model_config = config[args.output_prefix]
  
    output_prefix = args.output_prefix

    forget_candidates = generate_candidates(args.data_dir, model_config, output_prefix, 'forget')
    retain_candidates = generate_candidates(args.data_dir, model_config, output_prefix, 'retain')
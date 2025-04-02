import argparse
import json
import yaml

def load_config(config_path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format in {config_path}: {e}")
        return None

def build_answer_mapping(datapre_config, model_name):
    """Builds a mapping from answer_key to model name."""
    method_answer_mapping = {}
    method_answer_mapping["model_answer_0"] = "Vanilla Model"
    models = datapre_config[model_name]
    for model_name, model_config in models.items():
        answer_key = model_config['answer_key']
        method_answer_mapping[answer_key] = model_name
    return method_answer_mapping

def initialize_results_mapping_bak(method_answer_mapping):
     """Initializes the results mapping structure."""
     return {value: {'forget': {'relevance': [], 'fluency': []}, 'retain': {'relevance': [], 'fluency': []}}
            for key, value in method_answer_mapping.items()}
def initialize_results_mapping(method_answer_mapping):
     """Initializes the results mapping structure."""
     return {value: {'forget': [], 'retain': []}
            for key, value in method_answer_mapping.items()}

def process_results_bak(results, results_mapping, method_answer_mapping, task_type):
    """Processes forget or retain results."""
    for result in results:
        for key, value in result.items():
            if key in method_answer_mapping and key != 'id':
                try:
                    model_name = method_answer_mapping[key]
                    results_mapping[model_name][task_type]['relevance'].append(value['relevance'])
                    results_mapping[model_name][task_type]['fluency'].append(value['fluency'])
                except KeyError as e:
                    print(f"Error processing {task_type} result with id {result.get('id', 'unknown')}: {e}")

def calculate_average_metrics_bak(results_mapping):
    """Calculates the average relevance and fluency for each model and task."""
    for key, value in results_mapping.items():
        for task in ['forget', 'retain']:
            for metric in ['relevance', 'fluency']:
                if value[task][metric]:
                    results_mapping[key][task][metric] = sum(value[task][metric]) / len(value[task][metric])
                else:
                    results_mapping[key][task][metric] = 0
    return results_mapping
def process_results(results, results_mapping, method_answer_mapping, task_type):
    """Processes forget or retain results."""
    for result in results:
        for key, value in result.items():
            if key in method_answer_mapping and key != 'id':
                try:
                    model_name = method_answer_mapping[key]
                    results_mapping[model_name][task_type].append(value)
                except KeyError as e:
                    print(f"Error processing {task_type} result with id {result.get('id', 'unknown')}: {e}")

def calculate_average_metrics(results_mapping):
    """Calculates the average relevance and fluency for each model and task."""
    for key, value in results_mapping.items():
        for task in ['forget', 'retain']:
            if value[task]:
                results_mapping[key][task] = sum(value[task]) / len(value[task])
                if task == "retain":
                    results_mapping[key][task] = results_mapping[key][task]
            else:
                results_mapping[key][task] = 0
    return results_mapping


def main():
    parser = argparse.ArgumentParser(description="Process model evaluation results.")
    parser.add_argument("--config", type=str, default="./config/datapre.yaml", help="Path to the datapre YAML config file.")
    parser.add_argument("--forget_results", type=str, default="../llama2-results-archived-aggregated/llama2-7b_kud_forget_candidates_evaluated1.json", help="Path to the forget results JSON file.")
    parser.add_argument("--retain_results", type=str, default="../llama2-results-archived-aggregated/llama2-7b_kud_retain_candidates_evaluated1.json", help="Path to the retain results JSON file.")
    parser.add_argument("--output", type=str, help="Path to save the processed results JSON file.", default="../llama2-results-archived-aggregated/llama2-7b_kud_1.json",)
    parser.add_argument("--model_name", type=str, default="llama2-7b_kud", help="Model name for the results file.")
    args = parser.parse_args()


    # Load configurations
    datapre_config = load_config(args.config)
    if not datapre_config:
        return

    # Build answer key mapping
    method_answer_mapping = build_answer_mapping(datapre_config, args.model_name)

    # Initialize the results mapping
    results_mapping = initialize_results_mapping(method_answer_mapping)

    # Load the results data
    try:
         with open(args.forget_results, 'r') as f:
             forget_results = json.load(f)
         with open(args.retain_results, 'r') as f:
             retain_results = json.load(f)

    except FileNotFoundError as e:
        print(f"Error opening results file {e}")
        return
    except json.JSONDecodeError as e:
         print(f"Error decoding json file {e}")
         return

    # Process forget and retain results
    process_results(forget_results, results_mapping, method_answer_mapping, 'forget')
    process_results(retain_results, results_mapping, method_answer_mapping, 'retain')


    # Calculate average metrics
    results_mapping = calculate_average_metrics(results_mapping)

    # Save the results
    with open(args.output, 'w') as f:
        json.dump(results_mapping, f, indent=4)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
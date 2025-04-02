import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from tqdm import tqdm
from utils import gpt4o_chat
import argparse

# os.environ['http_proxy'] = 'http://127.0.0.1:20172'
# os.environ['https_proxy'] = 'http://127.0.0.1:20172'

with open("config/relev_fluen_prompt.txt", "r") as f:
    prompt_template = f.read()

def evaluate_single_case(case: Dict[str, Any]) -> Dict[str, Any]:
    # json dict to string
    case = str(case)
    query = prompt_template.replace("<DATA>", case)
    llm_response = gpt4o_chat(query)
    try:
        evaluation = json.loads(llm_response.replace('\n',''))
    except json.JSONDecodeError:
        print(f"JSONDecodeError: {llm_response}")
        evaluation = {"error": llm_response}
    return evaluation

def evaluate_cases_concurrently(data: list, max_workers: int) -> list:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(evaluate_single_case, data), total=len(data), desc="Evaluating"))
    return results

def entail_fluent_gpt4o(data_path, max_workers, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    evaluation_results = evaluate_cases_concurrently(data, max_workers)

    # for result in evaluation_results:
    #     print(json.dumps(result, indent=2))
    # Save the results to a file
    with open(save_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../kud-llama-results/llama2-7b_kud_forget_candidates.json")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="../kud-llama-gpt/llama2-7b_kud_forget_candidates_evaluated.json")
    args = parser.parse_args()

    max_workers = 10  # You can adjust this based on your system and API rate limits
    entail_fluent_gpt4o(args.data_path, args.max_workers, args.save_path)

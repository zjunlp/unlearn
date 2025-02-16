from pathlib import Path
import json
from utils import split_text, create_payload, invoke_llm_and_parse_response, merge_payloads_by_idx, merge_payload_text_chunks, remove_none_response, check_results, load_json
from copy import deepcopy
import concurrent.futures
import argparse

question_payload_template = {"idx": None, "text": None, "prompt": None, "variant_type": None, "response": None, "model": None}
text_payload_template = {"idx": None, "text": None, "part": None, "prompt": None, "variant_type": None, "response": None, "model": None}

#load templates
with open("templates.json", "r") as f:
    templates = json.load(f)
# create temp folder if not exists
Path("temp").mkdir(parents=True, exist_ok=True)

def process_qa(data_path: str, model:str, max_workers=8):
    data = load_json(data_path)

    data = [{'idx': idx, **d} for idx, d in enumerate(data)]

    processed_data = []
    
    # create payload for question variants ...
    question_payloads = []

    for i, item in enumerate(data):
        question = item[text_column]

        question_payload = deepcopy(question_payload_template)
        question_payload['idx'] = i
        question_payload['text'] = question
        payloads = create_payload(question_payload, templates, model, template_field="question_variants")
        question_payloads.extend(payloads)

    print("number of question payloads: ", len(question_payloads))
    # invoke llm and parse response for question variants (async pool)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        question_results = executor.map(invoke_llm_and_parse_response, question_payloads)
        question_results = list(question_results)
    question_results = remove_none_response(question_results)
    print("done question request")

    question_results_dict = merge_payloads_by_idx(question_results)
    # with open("temp/question_results.json", "w", encoding="utf-8") as f:
    #     json.dump(question_results_dict, f, indent=2, ensure_ascii=False)

    # process answer variants
    passed_idx_v = {}
    passed_results_list = []
    for _ in range(3):
        text_payloads = []
        for item in data:
            answer = item[label_column]
            idx = item['idx']
            questions = []
            # original question
            questions.append(data[idx][text_column])
            # question variants
            questions.extend(question_results_dict[idx]['response'])
            for qid, q in enumerate(questions):
                blocks = split_text(answer, strategy="length", chunk_size=800)
                for j, block in enumerate(blocks):
                    text_payload = deepcopy(text_payload_template)
                    text_payload['idx'] = idx
                    text_payload['text'] = block
                    text_payload['part'] = j
                    text_payload["query"] = q
                    text_payload["qid"] = qid
                    payloads = create_payload(text_payload, templates, model, template_field="text_variants", passed_idx_v=passed_idx_v)
                    text_payloads.extend(payloads)

        print("number of text payloads: ", len(text_payloads))
        if len(text_payloads) == 0:
            break
        # invoke llm and parse response for answer variants (async pool)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            text_results = executor.map(invoke_llm_and_parse_response, text_payloads)
            text_results = list(text_results)
        text_results = remove_none_response(text_results)
        print("done create request")

        # with open("temp/text_results.json", "w", encoding="utf-8") as f:
        #     json.dump(text_results, f, indent=2, ensure_ascii=False)

        text_results_ = deepcopy(text_results)

        # Update 'text' field 
        for payload in text_results:
            payload['text'] = payload['response']

        text_stage_check_payloads = []
        for payload in text_results:
            payloads = create_payload(payload, templates, model, template_field="text_check", passed_idx_v=passed_idx_v)
            text_stage_check_payloads.extend(payloads)
        
        print("number of text stage check payloads: ", len(text_stage_check_payloads))
        # invoke llm and parse response for misleading text variants (async pool)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            text_stage_check_results = executor.map(invoke_llm_and_parse_response, text_stage_check_payloads)
            text_stage_check_results = list(text_stage_check_results)
        text_stage_check_results = remove_none_response(text_stage_check_results)
        print("done text stage check request")

        # check if the response is correct
        passed_results, passed_iv = check_results(text_results_, text_stage_check_results)

        # update passed_idx_v
        for idx, v in passed_iv.items():
            if idx not in passed_idx_v:
                passed_idx_v[idx] = v
            else:
                passed_idx_v[idx].extend(v) 
        
        passed_results_list.extend(passed_results)

    # merge dicts by idx
    text_results = merge_payload_text_chunks(passed_results_list)

    text_results_dict = merge_payloads_by_idx(text_results)

    # with open("temp/text_results.json", "w", encoding="utf-8") as f:
    #     json.dump(text_results_dict, f, indent=2, ensure_ascii=False)

    for i in range(len(data)):
        original_question = data[i][text_column] 
        if i in question_results_dict:
            question_variants = question_results_dict[i]['response']
        else:
            question_variants = None
        original_answer = data[i][label_column]
        if i in text_results_dict:
            answer_variants = text_results_dict[i]['response']
        else:
            answer_variants = None

        # Save the processed question and answer variants in a reasonable format
        processed_data.append({
            "q_id": i,
            "original_question": original_question,
            "question_variants": question_variants,
            "original_answer": original_answer,
            "answer_variants": answer_variants
        })
    
    return processed_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/TOFU/forget10.jsonl", help="Path to the data file")
    parser.add_argument("--model", type=str, default="zhipu", help="Model to use")
    args = parser.parse_args()

    data_path = args.data_path
    model = args.model
    if "tofu" in data_path.lower():
        text_column = "question"
        label_column = "answer"
    else:
        text_column = "text"
        label_column = "labels"
    if Path(data_path).suffix == ".json" or Path(data_path).suffix == ".jsonl":
        results = process_qa(data_path, model)
    else:
        raise ValueError("Unsupported data format")

    with open("temp/results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
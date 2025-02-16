from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from peft import AutoPeftModelForCausalLM

templates = {"llama2": {"question_start_tag": "[INST] ","question_end_tag": ' [/INST]', "answer_tag": ""}, "llama3": {"question_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n","question_end_tag": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", "answer_tag": ""},  "gemma2": {"question_start_tag": "<start_of_turn>", "question_end_tag": "<end_of_turn>", "answer_tag": ""}, "default": {"question_start_tag": "", "question_end_tag": "", "answer_tag": ""}}

def eval(model_path, model, eval_data, tokenizer, output_file, device, use_vllm=False):
    results = []
    if "llama2" in model_path.lower() and "tofu" in model_path.lower():
        template = templates["llama2"]
    elif "llama3" in model_path.lower() and "tofu" in model_path.lower():
        template = templates["llama3"]
    elif "gemma" in model_path.lower() and "tofu" in model_path.lower():
        template = templates["gemma"]
    else:
        template = templates["default"]

    ignore_eos = False

    question_start_tag = template["question_start_tag"]
    question_end_tag = template["question_end_tag"]
    answer_tag = template["answer_tag"]
    if "tofu" in model_path.lower():
        text_column = "question"
        labels_column = "answer"
    else:
        text_column = "text"
        labels_column = "labels"
    
    if use_vllm:
        from vllm import LLM, SamplingParams
        max_iterations = 3
        iteration = 0

        for sample in eval_data:
            results.append({
                "query": question_start_tag + sample[text_column] + question_end_tag ,
                'ground_truth': sample[labels_column],
                'generated_response': ""
            })

        while True:
            iteration += 1
            unfinished_samples= [sample for sample in results if sample["generated_response"] == ""]

            if not unfinished_samples or iteration > max_iterations:
                break  
            querys = [sample["query"] for sample in unfinished_samples]

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=5,
                max_tokens=128,
                ignore_eos=ignore_eos  
            )
            try:
                outputs = model.generate(querys, sampling_params)
                
                for output in outputs:
                    generated_text = output.outputs[0].text
                    for i, sample in enumerate(results):
                        if output.prompt == sample["query"] and generated_text != "":
                            results[i]["generated_response"] = generated_text
                            break
            except Exception as e:
                print(f"An error occurred during generation: {e}")
                break  
    else:
        for sample in tqdm(eval_data):
            query = question_start_tag + sample[text_column]  + question_end_tag
            inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=256)
            
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            with torch.no_grad():  
                outputs = model.generate(
                    **inputs,
                    max_length=512,  
                    num_return_sequences=1,  
                    do_sample=True, 
                    top_p=0.9, 
                    top_k=5,  
                    temperature=0.7  
                )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = {
                'query': query,
                'ground_truth': sample[labels_column],
                'generated_response': generated_text
            }
            results.append(result)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, )
    parser.add_argument("--tokenizer_path",type=str)
    parser.add_argument("--forget_val_data_path", type=str,)
    parser.add_argument("--retain_val_data_path", type=str,)
    parser.add_argument("--output_file_forget", type=str,)
    parser.add_argument("--output_file_retain", type=str,)
    parser.add_argument("--use_vllm", action="store_true", default=False)

    args = parser.parse_args()
    if args.tokenizer_path is None:
        tokenizer_path = args.model_path
    else:
        tokenizer_path = args.tokenizer_path
    model_path = args.model_path  
    forget_val_data_path = args.forget_val_data_path
    retain_val_data_path = args.retain_val_data_path

    use_vllm = args.use_vllm  

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if 'llama' in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token  

    if use_vllm:
        from vllm import LLM, SamplingParams
        print(model_path, tokenizer_path)
        llm = LLM(model=model_path, tokenizer=tokenizer_path, gpu_memory_utilization=0.88, dtype='float16')
        model = llm  
        device = None  
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if Path(model_path).joinpath("adapter.json").exists():
            model = AutoPeftModelForCausalLM.from_pretrained(model_path).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    with open(args.forget_val_data_path, 'r') as f:
        if "tofu" in args.forget_val_data_path.lower():
            forget_val_data = [json.loads(line) for line in f]
        else:
            forget_val_data = json.load(f)

    with open(args.retain_val_data_path, 'r') as f:
        if "tofu" in args.retain_val_data_path.lower():
            retain_val_data = [json.loads(line) for line in f]
        else:
            retain_val_data = json.load(f)
    

    output_file_forget = args.output_file_forget
    output_file_retain = args.output_file_retain

    eval(model_path, model, forget_val_data, tokenizer, output_file_forget, device, use_vllm=use_vllm)
    eval(model_path, model, retain_val_data, tokenizer, output_file_retain, device, use_vllm=use_vllm)

    print(f"Results saved to {output_file_forget} and {output_file_retain}")
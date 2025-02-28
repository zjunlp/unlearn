from typing import List
import re
from copy import deepcopy
import json
from zhipuai import ZhipuAI
from openai import OpenAI

def load_json(file_path:str)->dict:
    """
    Load the JSON file and jsonl file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        try: # json format
            data = json.load(f)
        except: # jsonlines format
            f.seek(0)   
            data = [json.loads(line) for line in f]
    return data

# ================== Variants Generation ==================
zhipu_client = ZhipuAI(api_key="YOUR KEY") # enter your APIKey
qwen_client = OpenAI(api_key="YOUR KEY", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",)   
deepseek_client = OpenAI(api_key="YOUR KEY", base_url="https://api.deepseek.com")

def llm_api(prompt:str, model:str)->List[str]:
    """
    Call the LLM API to generate
    """
    messages = [
        {
        "role": "user",
        "content": prompt
        }
    ]
    if model == "zhipu":
        try:
            response = zhipu_client.chat.completions.create(
                model="glm-4-plus",  
                messages=messages,
                temperature=0.8,  
            )
            response = response.choices[0].message.content
        except Exception as e:
            response = None
    elif model == "qwen":
        try:
            completion = qwen_client.chat.completions.create(
                model="qwen-plus", # https://help.aliyun.com/zh/model-studio/getting-started/models
                messages=messages,
                )
            response = completion.choices[0].message.content
        except Exception as e:
            response = None
    elif model == "deepseek":
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            response = response.choices[0].message.content
        except Exception as e:
            response = None
    return response

def parse_response_text(response:str)->str:
    """
    Parse the response text
    """
    # TODO: Implement the response text parser
    if response is None:
        return None
    return response


def create_payload(payload, templates, model, template_field="question_variants", passed_idx_v=None):
    ret = []
    for variant_type, template in templates[template_field].items():
        new_payload = deepcopy(payload)
        if "qid" in new_payload:
            variant_type += f"_{new_payload['qid']}"
        if "check" in variant_type:
            new_payload['variant_type'] = new_payload["variant_type"]
        else:
            new_payload['variant_type'] = variant_type
        
        # Skip the idx that has been passed
        if passed_idx_v is not None:
            if new_payload['idx'] in passed_idx_v.keys() and new_payload["variant_type"] in passed_idx_v[new_payload['idx']]:
                continue
        if "query" in new_payload:
            new_payload['prompt'] = template.format(query=new_payload['query'], text=new_payload['text'])
        else:
            new_payload['prompt'] = template.format(query=new_payload['text'])
        new_payload['model'] = model
        ret.append(new_payload)
    return ret

def invoke_llm_and_parse_response(payload):
    max_retry = 3
    retry = 0
    while retry < max_retry:
        response = llm_api(payload['prompt'], payload["model"])
        if response is None:
            retry += 1
        else:
            break
    response_text = parse_response_text(response)
    payload['response'] = response_text
    return payload

def merge_payloads_by_idx(payloads):
    merged_dict = {}
    for payload in payloads:
        idx = payload['idx']
        if idx not in merged_dict:
            merged_dict[idx] = {}
            for k, v in payload.items():
                merged_dict[idx][k] = [v]
        else:
            for k, v in merged_dict[idx].items():
                merged_dict[idx][k].append(payload[k])
    return merged_dict

def remove_none_response(payloads):
    if not 'part' in payloads[0]:
        return [p for p in payloads if p['response'] is not None]
    # remove all chunks if any of the chunks is None
    else:
        ind_to_remove = set()
        for payload in payloads:
            ind = (payload['idx'], payload['variant_type'], )
            if payload['response'] is None:
                ind_to_remove.add(ind)
        return [p for p in payloads if (p['idx'], p['variant_type']) not in ind_to_remove]

def check_results(org_results, check_results):
    """
    Check the results of the data augmentation
    """
    # Create a lookup dictionary for faster access
    lookup = {}
    for check in check_results:
        key = (check['idx'], check['part'], check['variant_type'])
        lookup[key] = check['response']
    
    passed_list = []
    passed_dict = {}
    
    for item in org_results:
        key = (item['idx'], item['part'], item['variant_type'])
        if key in lookup:
            response = lookup[key]
            # Check if the last five letters, lowercase, contain 'no'
            if 'no' in response[-5:].lower():
                passed_list.append(item)
                idx = item['idx']
                variant_type = item['variant_type']
                if idx in passed_dict:
                    passed_dict[idx].append(variant_type)
                else:
                    passed_dict[idx] = [variant_type]
    
    return passed_list, passed_dict

def split_text_by_sentences(text:str)->List[str]:
    sentence_endings = r'(?<=[.!?]) +'
    sentences = re.split(sentence_endings, text)
    return sentences

def split_text_by_paragraphs(text:str)->List[str]:
    paragraphs = text.split("\n\n") 
    return [para.strip() for para in paragraphs if para.strip()]  

def split_text_by_length(text:str, chunk_size=500)->List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def split_text(text, strategy="paragraphs", chunk_size=500):
    if strategy == "sentences":
        return split_text_by_sentences(text)
    elif strategy == "paragraphs":
        return split_text_by_paragraphs(text)
    elif strategy == "length":
        return split_text_by_length(text, chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def merge_payload_text_chunks(payloads):
    merged_dict = {}
    for d in payloads:
        idx = d.get('idx')
        type_ = d.get('variant_type')
        part = d.get('part')
        text = d.get('text')
        response = d.get("response")

        key = (idx, type_)
        if key not in merged_dict:
            merged_dict[key] = deepcopy(d) 
            merged_dict[key]['part'] = {}
        
        if part not in merged_dict[key]['part']:
            merged_dict[key]['part'][part] = {'part': part, 'text': text, 'response': response}
    
    for v in merged_dict.values():
        dicts = list(v['part'].values())
        sorted_dicts = sorted(dicts, key=lambda x: x['part'])

        result_text = ''
        result_response = ''

        for d in sorted_dicts:
            result_text += d['text']
            result_response += d['response']
        v['response'] = result_response
        v['text'] = result_text
    
    for key in merged_dict.keys():
        del merged_dict[key]['part']

    return list(merged_dict.values())

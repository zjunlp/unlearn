from typing import List
import re
from copy import deepcopy
from openai import OpenAI
from pydantic import BaseModel
from typing import List
import json

class ModelAnswer(BaseModel):
    relevance: int
    fluency: int

class ResponseScore(BaseModel):
    id: str
    model_answer_0: ModelAnswer
    model_answer_1: ModelAnswer
    model_answer_2: ModelAnswer
    model_answer_3: ModelAnswer
    model_answer_4: ModelAnswer
    model_answer_5: ModelAnswer
    model_answer_6: ModelAnswer
    model_answer_7: ModelAnswer
    model_answer_8: ModelAnswer
    model_answer_9: ModelAnswer

def dpsk_chat(prompt:str)->List[str]:
    client = OpenAI(api_key="YOUR DeepSeek API", base_url="https://api.deepseek.com")

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=512, # more than 256 tokens
            stream=False
        )
        json_str = response.choices[0].message.content
        start = json_str.find('{')
        end = json_str.rfind('}')

        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        return json.dumps(json.loads(json_str))
        
    except json.JSONDecodeError as je:
        print(f"JSON decode error: {str(je)}")
        print(f"response: {json_str}")
        return json.dumps({"error": "Failed to parse JSON response"})
    except Exception as e:
        print(f"API error: {str(e)}")
        return json.dumps({"error": str(e)})

def gpt4o_chat(prompt:str)->List[str]:
    client = OpenAI(api_key="YOUR KEY")

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            response_format=ResponseScore,
            max_tokens=256,
        )
    except Exception as e:
        response = None
        output = str(e)

    if response is not None:
        output = response.choices[0].message.content
    else:
        print(f"Error: {output}")
        pass
    return output

def parse_response_text(response:str)->str:
    """
    Parse the response text
    """
    # TODO: Implement the response text parser
    if response is None:
        return None
    return response


def create_payload(payload, templates, model, template_field="question_variants"):
    ret = []
    for variant_type, template in templates[template_field].items():
        new_payload = deepcopy(payload)
        new_payload['variant_type'] = new_payload["variant_type"] + "__" + variant_type if new_payload["variant_type"] else variant_type
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


# ================== Text Splitting ==================
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

# ================== TODO:Text filter ==================
import argparse
import json
from typing import List, Dict, Callable
import re
import numpy as np
import torch
import nltk
from nltk.tokenize import word_tokenize
import math
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..dataAugment.utils import llm_api
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random

payload_template = {"model": "deepseek"}
prompt_template = '''>>>query:{query}\nresponse:{response}<<<

Extract key entities from the response (excluding those already in the query):
1. Specific entities: name*, email*, locations*, dates*, organizations, events, technical terms
2. Core nouns from noun phrases: prefer extracting only the main noun (e.g., "literary" from "literary projects")
3. Only return the single core word when it's multi-word entity phrases

Avoid extracting common verbs or general defination(like 'email', 'people', 'events' 'books' and so on)

Return a list of unique entities as comma-separated values (duplicates should appear only once), without additional explanations.'''

def filter_entities(query: str, entities: list[str]) -> list[str]:
    query_words = set(word_tokenize(query.lower()))
    
    # Filter entities that don't have word overlap with query
    filtered = [
        entity for entity in entities 
        if not query_words & set(word_tokenize(entity.lower()))
    ]
    return filtered

def cosine_similarity_matrix(embeddings1, embeddings2):
    norm_embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm_embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    return np.dot(norm_embeddings1, norm_embeddings2.T)

def fuzzy_jaccard_similarity_from_embeddings(embeddings1, reference, threshold=0.8):
    cos_sim_matrix = cosine_similarity_matrix(embeddings1, reference)
    
    matches = np.where(cos_sim_matrix >= threshold)
    
    matched_pairs = set(zip(matches[0], matches[1]))
    
    intersection_size = len(matched_pairs)
    #union_size = embeddings1.shape[0] + reference.shape[0] - intersection_size
    union_size = reference.shape[0]
    return intersection_size / union_size if union_size else 1, cos_sim_matrix.tolist()

def parse_response(response) -> set:
    """Parse the response from the LLM API."""
    if not isinstance(response, str):
        return None
    response = response.split(",")
    response = [r.strip(" \n\\/\"") for r in response if r]
    return set(response)

def process_payload(payload):
    """Wrapper for processing a single prompt with the LLM API."""
    max_try = 3
    payload["pred_entity"] = set()
    payload["gt_entity"] = set()
    for _ in range(max_try):
        response = llm_api(payload['pred_prompt'], payload['model'])
        response = parse_response(response)
        if response is not None:
            payload['pred_entity'] = response
            break
    for _ in range(max_try):
        response = llm_api(payload['gt_prompt'], payload['model'])
        response = parse_response(response)
        if response is not None:
            payload['gt_entity'] = response
            break
    return payload


def eval_rouge_recall(gen_outputs, ground_truths):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = []
    rougeL_recall = []
    info = []
    for idx, (gen, gt) in enumerate(zip(gen_outputs, ground_truths)):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall.append(rouge_scores['rouge1'].recall)
        rougeL_recall.append(rouge_scores['rougeL'].recall)
        info.append({"idx": idx, "pred": gen, "gt": gt, "r1r": rouge_scores['rouge1'].recall, "rLr": rouge_scores['rougeL'].recall})
    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}, info


def get_entailment_results(pipe, queries, gen_outputs, ground_truths, bs=1, forget=True):
    results = []
    for i in range(0, len(gen_outputs)):
        r_ = {"pred": gen_outputs[i], "gt": ground_truths[i], "query": queries[i]}
        targets = [ground_truths[i]]
        outputs = [gen_outputs[i]]
        if forget:
            data_list = [{'text': outputs[i], 'text_pair': targets[i]} for i in range(len(targets))]
        else:
            data_list = [{'text': targets[i], 'text_pair': outputs[i]} for i in range(len(targets))]
        r_.update(pipe(data_list)[0])
        results.append(r_)

    return results

def similarity_score(predicted: str, reference: str, similarity_model: Callable):
    all_sentences = predicted + reference

    embeddings = similarity_model.encode(all_sentences, convert_to_tensor=False) 

    predicted_embeddings = embeddings[:len(predicted)]
    reference_embeddings = embeddings[len(predicted):]

    return predicted_embeddings, reference_embeddings


def extract_entities(text: str, entity_extractor: Callable) -> set:
    doc = entity_extractor(text)
    return {ent.text for ent in doc.ents}

def extract_entities_v2(data, entity_extractor: Callable) -> List[Dict]:
    payloads = []
    for d in data:
        payload = deepcopy(payload_template)
        payload.update(**d)
        pred_prompt = deepcopy(prompt_template)
        pred_prompt = pred_prompt.format(query=payload['query'], response=payload["pred"])
        gt_prompt = deepcopy(prompt_template)
        gt_prompt = gt_prompt.format(query=payload["query"], response=payload['gt'])
        payload["pred_prompt"] = pred_prompt
        payload["gt_prompt"] = gt_prompt
        payloads.append(payload)
    
    # Use ThreadPoolExecutor with tqdm
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Wrap executor.map with tqdm to show progress
        results = list(tqdm(executor.map(process_payload, payloads), total=len(payloads)))
    
    # calculate entity similarity
    for item in results:
        predicted = list(item["pred_entity"])
        predicted = filter_entities(item["query"], predicted)
        reference = list(item["gt_entity"])
        if len(predicted) == 0 or len(reference) == 0:
            item["entity_sim"] = 0
            continue
        predicted_embeddings, reference_embeddings = similarity_score(predicted, reference, entity_extractor)
        item["entity_sim"], item["sim_mat"] = fuzzy_jaccard_similarity_from_embeddings(predicted_embeddings, reference_embeddings)
    
    return results

def compute_ppl(query: str, response: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    input_ids = tokenizer(
                    query,
                    return_tensors='pt'
                ).input_ids.to(model.device)
    combined_input_ids = tokenizer(
        query+response,
        return_tensors='pt'
    ).input_ids.to(model.device)
    combined_target_ids = combined_input_ids.clone()
    combined_target_ids[:,:len(input_ids[0])] = -100
    try:
        with torch.no_grad():
            outputs = model(combined_input_ids, labels=combined_target_ids)
            loss = outputs.loss.item()
        return torch.exp(torch.tensor(loss)).item()
    except:
        return -1

def calculate_rttr(text: str) -> float:
    tokens = word_tokenize(text.lower())
    unique_tokens = set(tokens)
    total_tokens = len(tokens)
    return len(unique_tokens) / math.sqrt(total_tokens) if total_tokens > 0 else 0

def calculate_BI(text: str) -> float:
    tokens = word_tokenize(text.lower())

    V = len(set(tokens))
    total_tokens = len(tokens)
    
    # Brunet's Index, https://linguistics.stackexchange.com/questions/27365/formula-for-brun%c3%a9ts-index
    if total_tokens > 0:
        BI = total_tokens ** (1 / (V ** 0.165))
    else:
        BI = 1000
    
    return BI

def calculate_honores_statistic(text: str, max_value=int(1e6), epsilon = 1e-1):
    words = word_tokenize(text.lower())

    N = len(words)
    if N <= 1:
        return 0
        
    U = len(set(words))
    if U == 0:
        return 0

    word_frequency = nltk.FreqDist(words)
    N_uni = sum(1 for word, freq in word_frequency.items() if freq == 1)

    HS = 100 * math.log(N) / (1 - N_uni / U + epsilon)

    return min(HS, max_value)

def compute_kfr(forget_queries, forget_answers: List[str], forget_origin_answers: List[str],coverage_model: Callable, entity_extractor: Callable, theta_1, pipe) -> float:
    forget_labels = []
    entity_scores = []
    kfr_info = []

    rouge_recall, rouge_info = eval_rouge_recall(forget_answers, forget_origin_answers)
    mean_rouge_recall = {k: np.mean(v) for k, v in rouge_recall.items()}
    entailment_results = get_entailment_results(pipe, forget_queries, forget_answers, forget_origin_answers, forget=True)
    entity_results = extract_entities_v2(entailment_results, coverage_model)
    for item in entity_results:
        query = item["query"]
        pred = item["pred"]
        gt = item["gt"]
        ent_label = item["label"]
        ent_score = item["score"]
        
        entities_gt = item["gt_entity"]
        entities_pred = item["pred_entity"]
        # d_entities = 1 - item["entity_sim"]
        # if len(entities_gt) <=3:
        #     flag = (d_entities >= 0.5)
        # else:
        #     flag = (d_entities >= theta_1)
        # entity_scores.append(flag)  
        d_entities = item["entity_sim"]
        if len(entities_gt) <=3:
            flag = (d_entities <= 0.5)
        else:
            flag = (d_entities <= theta_1)
        entity_scores.append(flag)  

        
        if ent_label == "contradiction" or flag:
            forget_label = 1
        else:
            forget_label = 0
        forget_labels.append(forget_label)
        
        kfr_info.append({"query":query, "pred": pred, "gt": gt, "ent_label": ent_label, "ent_score":ent_score, "forget_label":forget_label, "entity_score": d_entities, "entity_label": d_entities >= theta_1, "entity_gt": list(entities_gt), "entity_pred": list(entities_pred), })
        
    rouge_info = [{k: v for k, v in item.items() if k != 'idx'} for item in rouge_info] if rouge_info else []

    return np.mean(forget_labels), mean_rouge_recall, rouge_info, kfr_info


def compute_krr(retain_queries, predicted_answers: List[str], gold_answers: List[str], coverage_model: Callable, entity_extractor: Callable, theta_2,  pipe) -> float:
    retention_labels = []
    krr_info = []

    rouge_recall, rouge_info = eval_rouge_recall(predicted_answers, gold_answers)
    mean_rouge_recall = {k: np.mean(v) for k, v in rouge_recall.items()}

    entailment_results = get_entailment_results(pipe, retain_queries, predicted_answers, gold_answers, forget=False)
    entity_results = extract_entities_v2(entailment_results, coverage_model)
    entity_scores = []
    for item in entity_results:
        query = item["query"]
        pred = item["pred"]
        gt = item["gt"]
        ent_label = item["label"]
        ent_score = item["score"]
        
        entities_gold = item["gt_entity"]
        entities_pred = item["pred_entity"]
        e_match = item["entity_sim"]
        entity_scores.append(e_match >= theta_2)
        
        if ent_label != "contradiction" and e_match >= theta_2:
            retain_label = 1
        else:
            retain_label = 0
        
        retention_labels.append(retain_label)
        krr_info.append({
            "query": query,
            "pred": pred,
            "gt": gt,
            "ent_label": ent_label,
            "ent_score": ent_score,
            "retention_label": retain_label,
            "entity_score": e_match,
            "entity_label": e_match >= theta_2,
            "entity_gt": list(entities_gold),
            "entity_pred": list(entities_pred),
        })

    rouge_info = [{k: v for k, v in item.items() if k != 'idx'} for item in rouge_info] if rouge_info else []

    return np.mean(retention_labels), mean_rouge_recall, rouge_info, krr_info


def compute_as(all_queries: List[str], all_responses: List[str], language_model: Callable, tokenizer: Callable) -> float:
    ppl_scores = []
    BI_scores = []
    HS_scores = []
    RTTR_scores = []
    info = []
    for query, response in tqdm(zip(all_queries, all_responses), desc="Calculating Answer Stability (AS)", total=len(all_responses)):
        ppl_score = compute_ppl(query, response, language_model, tokenizer)
        if not np.isnan(ppl_score) and ppl_score != -1:
            ppl_scores.append(ppl_score)
        else:
            print(f"Warning: PPL score is NaN for response: {response}")
        rttr_score = calculate_rttr(response)
        RTTR_scores.append(rttr_score)
        bi_score = calculate_BI(response)
        if bi_score != 0:
            BI_scores.append(bi_score)
        HS_score = calculate_honores_statistic(response)
        HS_scores.append(HS_score)

        info.append({"response": response, "ppl": ppl_score, "ttr": rttr_score, "BI": bi_score, "HS": HS_score})
    
    return np.mean(ppl_scores), np.mean(RTTR_scores), np.mean(BI_scores), np.mean(HS_scores), info

def sigmoid(x, k=1, b=0):
  return 1 / (1 + np.exp(-k * x + b))

def as_mean(ppl_score, BI_scores, HS_scores) -> float:
    ppl_modified = sigmoid(-np.log(ppl_score), 1, 0)
    bi_modified =  sigmoid(-np.log(BI_scores), 1, 0)
    hs_modified = sigmoid(np.log(HS_scores), 1, 0)

    # harmonic mean
    harmonic_as_score = 3 / (1/ppl_modified + 1/bi_modified + 1/hs_modified)
    # arithmetic mean
    arithmetic_as_score = (ppl_modified + bi_modified + hs_modified) / 3
    
    return harmonic_as_score, arithmetic_as_score

def evaluate_models(forget_queries: List[str], forget_origin_answers: List[str], forget_answers: List[str], retain_queries: List[str], retain_answers: List[str],retain_gold_answers: List[str], semantic_model: Callable, entity_extractor: Callable, pipe: Callable, language_model: Callable, tokenizer: Callable,theta_1: float, theta_2: float,) -> Dict[str, float]:
    kfr, rouge_recall_kfr, rouge_info_kf,ent_info_f = compute_kfr(forget_queries,forget_answers, forget_origin_answers, semantic_model,entity_extractor, theta_1,pipe)
    rougeL_recall_kfr = rouge_recall_kfr['rougeL_recall']
    krr, rouge_recall_krr, rouge_info_kr,ent_info_r = compute_krr(retain_queries, retain_answers, retain_gold_answers, semantic_model,entity_extractor, theta_2, pipe)
    rougeL_recall_krr = rouge_recall_krr['rougeL_recall']


    ppl_score_f, ttr_scores_f, BI_score_f, HS_score_f, as_info_f = compute_as(forget_queries, forget_answers, language_model, tokenizer)
    ppl_score_r, ttr_scores_r, BI_score_r, HS_score_r, as_info_r = compute_as(retain_queries, retain_answers, language_model, tokenizer)

    # compute the aggregate results
    # 1. rougeL & kfr  mean
    positive_rougeL_f = 1 - rougeL_recall_kfr
    harmonic_forget_score = 2 * positive_rougeL_f * kfr / (positive_rougeL_f + kfr) if positive_rougeL_f + kfr > 0 else 0
    mean_forget_score = (positive_rougeL_f + kfr) / 2
    # 2. rougeL & krr mean
    harmonic_retain_score = 2 * rougeL_recall_krr * krr / (rougeL_recall_krr + krr) if rougeL_recall_krr + krr > 0 else 0
    mean_retain_score = (rougeL_recall_krr + krr) / 2

    # 3. as mean
    harmonic_as_score_f, arithmetic_as_score_f = as_mean(ppl_score_f, BI_score_f, HS_score_f)
    harmonic_as_score_r, arithmetic_as_score_r = as_mean(ppl_score_r, BI_score_r, HS_score_r)

    results = {"RougeL_recall_F": rougeL_recall_kfr, "RougeL_recall_R": rougeL_recall_krr, "KFR": kfr, "KRR": krr,
               "PPL_F": ppl_score_f, "PPL_R": ppl_score_r,"TTR_F": ttr_scores_f,"TTR_R": ttr_scores_r ,"BI_F": BI_score_f,  "BI_R":BI_score_r, "HS_F": HS_score_f, "HS_R": HS_score_r, "Harmonic_F":harmonic_forget_score, "Harmonic_R":harmonic_retain_score, "Mean_F":mean_forget_score, "Mean_R":mean_retain_score, "Harmonic_AS_F": harmonic_as_score_f, "Harmonic_AS_R": harmonic_as_score_r, "Arithmetic_AS_F": arithmetic_as_score_f, "Arithmetic_AS_R": arithmetic_as_score_r} 
    info = {"rouge_info_F": rouge_info_kf, "rouge_info_R": rouge_info_kr, "as_info_F": as_info_f, "as_info_R": as_info_r, "ent_info_f":ent_info_f, "ent_info_r":ent_info_r}
    return results, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_model_path", type=str, default="../Meta-Llama-3-8B-Instruct")
    parser.add_argument("--embedding_model_path", type=str, default="../all-MiniLM-L12-v2")
    parser.add_argument("--entailment_model_path", type=str, default="../deberta-v3-base-tasksource-nli")
    parser.add_argument("--test_model_name", type=str, default="llama3-8b")
    parser.add_argument("--forget_path", type=str, default="../evals/llama3/jsons/checkpoint-1500_gen_forget.json")
    parser.add_argument("--retain_path", type=str, default="../evals/llama3/jsons/checkpoint-1500_gen_retain.json")
    parser.add_argument("--output_path", type=str, default="results.json")
    parser.add_argument("--theta_1", type=float, default=0.3)
    parser.add_argument("--theta_2", type=float, default=0.3)

    args = parser.parse_args()
    # entity model
    entity_extractor = None
    # language model
    model_name = args.language_model_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
    # entailment model
    pipe = pipeline('text-classification', model=args.entailment_model_path, device=0)
    semantic_model = SentenceTransformer(args.embedding_model_path, ).to("cuda:0") 

    theta_1 = args.theta_1
    theta_2 = args.theta_2

    random.seed(42)
    with open(args.forget_path, 'r') as f:
        forgetdata = json.load(f)
        if 'tofu' in args.forget_path:
            forgetdata = random.sample(forgetdata, min(200, len(forgetdata)))
    forget_queries = [item['query'] for item in forgetdata]
    forget_origin_answers = [item['ground_truth'] for item in forgetdata]
    forget_answers = [item['generated_response'] for item in forgetdata]

    with open(args.retain_path, 'r') as f:
        retaindata = json.load(f)
        if 'tofu' in args.retain_path:
            retaindata = random.sample(retaindata, min(200, len(retaindata)))
    retain_queries = [item['query'] for item in retaindata]
    retain_answers = [item['generated_response'] for item in retaindata]
    retain_gold_answers = [item['ground_truth'] for item in retaindata]

    results, info = evaluate_models(forget_queries, forget_origin_answers, forget_answers, retain_queries, retain_answers, retain_gold_answers,semantic_model,entity_extractor, pipe, model, tokenizer, theta_1, theta_2,)
    for key, value in results.items():
        if isinstance(value, np.float32):
            results[key] = float(value)
    results["Model"] = args.test_model_name

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    with open(args.output_path.replace(".json", "_info.json"), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
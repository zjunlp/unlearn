import json
import argparse
import re
import random
from copy import deepcopy
from datasets import load_dataset
from pathlib import Path

def gather(data, text_column, labels_column):
    new_results = []

    for item in data:
        new_result = []
        length = min(len(item['question_variants']), len(item['answer_variants']))
        new_result.append({
            text_column: item['original_question'],
            labels_column: item['original_answer'],
        })
        for i in range(length):
            new_result.append({
                text_column: item['question_variants'][i],
                labels_column: item['answer_variants'][i]
            })
        new_results.extend(new_result)
    return new_results



def contains_chinese(text):
    # check if the text contains Chinese characters
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def chinese_ratio(text):
    # check the ratio of Chinese characters in the text
    if not text:
        return 0
    chinese_count = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(text.replace(" ", ""))  
    return chinese_count / max(1, total_chars)  

def filter_and_clean(sentences, text_column, labels_column, threshold=0.01):
    cleaned_sentences = []
    for sentence in sentences:
        text = sentence[text_column]
        labels = sentence[labels_column]
        labels_ratio = chinese_ratio(labels)
        text_ratio = chinese_ratio(text)
        ratio = max(labels_ratio, text_ratio)
        if ratio > threshold:
            # if the ratio of Chinese characters is higher than the threshold, skip
            continue
        else:
            # remove Chinese characters
            cleaned_labels = re.sub(r'[\u4e00-\u9fff]', '', labels)
            cleaned_text = re.sub(r'[\u4e00-\u9fff]', '', text)
            cleaned_sentences.append({
                text_column: cleaned_text,
                labels_column: cleaned_labels
            })
    return cleaned_sentences

def cut(data, text_column, labels_column):
    new_data = []
    for d in data:
        answer = d[labels_column]
        answer = answer.split(" ")
        # cut answer 25% 50% 75%
        for i in range(1, 4):
            if i != 1:
                # you can try different cut ratios, but here we only cut 25% here
                continue
            new_d = deepcopy(d)
            new_d[labels_column] = " ".join(answer[int(len(answer) * i / 4):])
            new_d[text_column] = " ".join(answer[:int(len(answer) * i / 4)])
            new_data.append(new_d)
    data.extend(new_data)
    return data

def add_wikiqa(data, text_column, labels_column, mix_ratio=1.2):
    wikiqa_subset = load_dataset("microsoft/wiki_qa",)
    wikiqa_subset = wikiqa_subset["train"].shuffle(seed=42+2017)
    wikiqa = []
    for item in wikiqa_subset:
        if item["label"] == 0:
            continue
        wikiqa.append({
            text_column: item["question"],
            labels_column: item["answer"]
        })
    # calculate the target wikiqa data length
    data_text_len = len(data)
    target_wikiqa_len = int(data_text_len * mix_ratio)
    
    # initialize wikiqa text length
    mixed_data = data

    wikiqa_text_len = 0
    
    # traverse the wikiqa subset until the target wikiqa text length is reached
    for wikiqa_text in wikiqa:
        mixed_data.append(wikiqa_text)
        wikiqa_text_len += 1
        if wikiqa_text_len >= target_wikiqa_len:
            break
    return mixed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../dataset/TOFU/forget10.jsonl", help="Path to the data file")
    parser.add_argument("--save_path", type=str, default="../dataset/augument_data/tofu.jsonl", help="Path to save the data file")
    args = parser.parse_args()

    if "tofu" in args.data_path.lower():
        text_column = "question"
        labels_column = "answer"
    else:
        text_column = 'text'
        labels_column = 'labels'

    # load the data
    with open("temp/results.json", "r") as f:
        data = json.load(f)
    
    # gather the data
    gathered_data = gather(data, text_column, labels_column)
    # shuffle the data
    random.shuffle(gathered_data)
    # filter and clean the data
    filtered_data = filter_and_clean(gathered_data, text_column, labels_column)

    # cut the data
    cut_data = cut(filtered_data, text_column, labels_column)

    # add wikiqa data
    final_data = add_wikiqa(cut_data, text_column, labels_column)

    # save the data
    # make sure the save_path parent directory exists
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    if "tofu" in args.data_path.lower():
        with open(args.save_path, "w", encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with open(args.save_path, "w", encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
import json

import pandas as pd

import re

from tqdm import tqdm

 

 

FILE_NAMES = [
    'cot_dataset_llama3_70_instruct.json',
    'cot_dataset_gemini_pro.json',
    'cot_dataset_gpt4.json',
    'cot_dataset_llama2_7_chat_hf.json',
    'cot_dataset_llama3_8b_instruct.json',
    'cot_dataset_qwen2_15b_instruct.json',
    'cot_dataset_qwen2_7b_instruct.json',
    'cot_dataset_qwen25_05b_instruct.json',

]

 

def format_prompt(context, question):
    return f"""Solve a question answering task by having a Thought, then Finish with your answer.
    Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
    You will be given context that you should use to help you answer the question.
 
    Relevant Context:
    {context}

    Question:
    {question}
    """


 

def is_correct(row, g_answer) :
    gold_answer = row["gold_answer"]
    gold_answer = 'Action: Finish[' + gold_answer + "]"

    if g_answer == gold_answer :
        return True
    else :
        return False



def process_dataset(file_name) :
    result = {}

    with open("cot_dataset_rm.json") as file_r :
        base_dataset = json.load(file_r)

    with open(file_name) as file_r :
        dataset = json.load(file_r)

    for i, idx in enumerate(dataset) :
        row = dataset[idx]
        g_thought = row['generated_trace']['thought']
        g_answer = row['generated_trace']['answer']

        # Check if idx already exists in the base dataset
        if idx in base_dataset :
            base_row = base_dataset[idx]

            if is_correct(row, g_answer) :
                base_row['c_response'].append(g_thought + g_answer)
            else:
                base_row['r_response'].append(g_thought + g_answer)
                
            base_row = {idx:base_row}
            result.update(base_row)
        else:

            if is_correct(row, g_answer) :
                record = {idx:{
                            "prompt": format_prompt(row["context"], row["question"]), #context + question
                            "gold_answer" : row["gold_answer"],
                            "level" : row["level"],
                            "type" : row["type"],
                            "c_response" : [g_thought + g_answer],
                            "r_response" : []
                        }}

            else :
                record = {idx:{
                            "prompt": format_prompt(row["context"], row["question"]), #context + question
                            "gold_answer" : row["gold_answer"],
                            "level" : row["level"],
                            "type" : row["type"],
                            "c_response" : [],
                            "r_response" : [g_thought + g_answer]
                        }}

            result.update(record)

    with open('cot_dataset_rm.json', "w") as out_file:

        out_file.write(json.dumps(result))   

    

    

def main() :

    for file_name in FILE_NAMES :
        process_dataset(file_name)

    with open("cot_dataset_rm.json") as file_r :
        base_dataset = json.load(file_r)

    result = []

    for i, idx in enumerate(base_dataset) :
        base_row = base_dataset[idx]
        c_responses = base_row['c_response']
        r_responses = base_row['r_response']
        list_responses = list(zip(c_responses, r_responses))

        for j in range(len(list_responses)):
            record = {
                'idx' : idx,
                'prompt' : base_row['prompt'],
                'gold_answer' : base_row['gold_answer'],
                'level' : base_row['level'],
                'type' : base_row['type'],
                'c_response' : list_responses[j][0],
                'r_response' : list_responses[j][1]
            }
            result.append(record)


    with open('cot_dataset_rm_final.json', "w") as out_file:
        out_file.write(json.dumps(result))
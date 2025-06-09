import sys
sys.path.append('..')
sys.path.append('../..')
root = '../root/'

import os
import json
import subprocess
from datasets import load_dataset
from tqdm import tqdm
from google import genai
from ollama import chat
import pandas as pd
import numpy as np
from llm import AnyOpenAILLM
import re
import argparse
from langchain.chat_models import ChatOpenAI
 

# CONFIG
NUM_SAMPLES = 2000  # how many QA pairs to generate
OLLAMA_MODEL = ['llama7b', 'llama70b', 'llama3_b', 'qwen0.5b', 'qwen1.5b' ,'qwen7b']

def format_prompt(context, question):

    return f"""Solve a question answering task by having a Thought, then Finish with your answer.
    Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.
    You will be given context that you should use to help you answer the question.

    Relevant Context:
    {context}

    Question:
    {question}

    Thought: <Your Thought here>

    Action: Finish[<Your Answer here>]
    """

 

def call_gemini(prompt: str, model: str) -> str :
    client = genai.Client(api_key=os.environ['OPENAI_API_KEY'])
    try:
        response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
        return response.text

    except Exception as e:
        print("Error from Gemini:", e)
        return ""

   

def call_ollama(prompt: str, model: str) -> str :
    response = chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

 

def call_gpt(prompt: str, model:str) -> str :

    
    llm = ChatOpenAI(
        model_name=model,
        temperature=0,
        max_tokens=250,
        api_key=os.environ['OPENAI_API_KEY'],  # Optional if set via environment
    )
    response = llm.predict(prompt)
    #print(response)
    return response


 
def get_answer(prompt: str, model_name: str) -> str :

    if model_name == 'gemini_pro':
        model_name = 'gemini-2.5-pro-preview-05-06'
        return call_gemini(prompt, model_name)
    
    elif model_name == 'gpt4':
        model_name = 'gpt-4o-mini'
        return call_gpt(prompt, model_name)
    
    elif model_name in OLLAMA_MODEL :
        return call_ollama(prompt, model_name)
   

def main() :
    parser = argparse.ArgumentParser(description ='Generate data')
    parser.add_argument('--model_name', type=str, required=True, choices=['gemini_pro', 'gpt4', 'llama7b', 'llama70b', 'llama3_b', 'qwen0.5b', 'qwen1.5b', 'qwen7b'])
    args = parser.parse_args()

    dataset = load_dataset("/mnt/c/Users/nayou/Personal/tulipe/reflexion/hotpotqa_runs/data/hotpot_qa.py", name="distractor")
    train_dataset = dataset["train"].to_pandas()[:2000]


    train_dataset['supporting_paragraphs'] = None
    
    for ind, row in train_dataset.iterrows():
        supporting_articles = row['supporting_facts']['title']
        articles = row['context']['title']
        sentences = row['context']['sentences']
        supporting_paragraphs = []

        for article in supporting_articles:
            supporting_paragraph = ''.join(sentences[np.where(articles == article)][0])
            supporting_paragraphs.append(supporting_paragraph)

        supporting_paragraphs = '\n\n'.join(supporting_paragraphs)
        train_dataset.at[ind, 'supporting_paragraphs'] = supporting_paragraphs

 
    with open(f"cot_dataset_{args.model_name}", "w") as out_file:

        result = {}
        for i, row in tqdm(train_dataset.iterrows(), total=min(NUM_SAMPLES, len(train_dataset))):
            if i >= NUM_SAMPLES:
                break

            question = row["question"]
            answer = row["answer"]
            context = row["supporting_paragraphs"]
            level = row["level"]
            type_r = row["type"]

            prompt = format_prompt(context, question)

            try:

                response = get_answer(prompt, args.model_name)
                pattern = r'(Action: )(Finish)\[(.+)\]'
                match = re.search(pattern, response)

                if match :
                    response = re.sub(pattern, '', response)

                    record = {row["id"] :{
                        "question": question,
                        "context": context,
                        "gold_answer": answer,
                        "level":level,
                        "type":type_r,
                        "generated_trace": {
                            "thought":response,
                            "answer":match.group()
                        }
                    }}

                    result.update(record)

            except Exception as e:
                print(f"Error at index {i}: {e}")

        out_file.write(json.dumps(result))
        
        
if __name__ == "__main__" : 
    main()
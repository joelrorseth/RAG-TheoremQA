import os
import torch
import json
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime

# Global gpt client
client = None

def gpt_setup():
    global client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise KeyError("OpenAI api key is not set.")
    client = OpenAI(api_key=api_key)


def run_tot_gpt(model_name, prompt, n_sample=1, stop='\n', temperature=0.7, max_tokens=1024):
    global client

    outputs = []
    completion = client.chat.completions.create(    
        model = model_name,
        messages = [
            {"role": "user", "content": prompt}
        ],
        n = n_sample, 
        stop = stop, 
        temperature = temperature, 
        max_tokens = max_tokens
    )
    outputs.extend([choice.message.content for choice in completion.choices])
    return outputs


def run_gpt3_5_turbo(entry, prompt_method):
    global client

    model_name = "gpt-3.5-turbo"
    # completion = client.chat.completions.create(
    #     model = model_name,
    #     messages=[
    #         {"role": "user", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease read a math problem, and then think step by step to derive the answer. The answer needs to be in numerical form.\n\n### Input:\nHow many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?\n\n### Response:."}
    #     ]
    # )
    # print(completion.choices[0].message)


def run_gpt4(entry, prompt_method):
    global client

    model_name = "gpt-4"
    # completion = client.chat.completions.create(
    #     model = model_name,
    #     messages=[
    #         {"role": "user", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease read a math problem, and then think step by step to derive the answer. The answer needs to be in numerical form.\n\n### Input:\nHow many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?\n\n### Response:."}
    #     ]
    # )
    # print(completion.choices[0].message)


    













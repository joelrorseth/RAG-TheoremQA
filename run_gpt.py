import torch
import json
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime


def run_gpt3_5_turbo(prompt: str):
    client = OpenAI()

    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease read a math problem, and then think step by step to derive the answer. The answer needs to be in numerical form.\n\n### Input:\nHow many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?\n\n### Response:."}
        ]
    )
    print(completion.choices[0].message)


    


def run_gpt4(prompt: str):
    client = OpenAI()

    completion = client.chat.completions.create(
        model = "gpt-4",
        messages=[
            {"role": "user", "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease read a math problem, and then think step by step to derive the answer. The answer needs to be in numerical form.\n\n### Input:\nHow many ways are there to divide a set of 8 elements into 5 non-empty ordered subsets?\n\n### Response:."}
        ]
    )
    print(completion.choices[0].message)












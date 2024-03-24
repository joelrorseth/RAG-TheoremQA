import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from run_tinyllama import run_TinyLlama


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='tinyllama', type=str) # Default later change to None

args = parser.parse_args()


def run_gpt3_prompt(prompt: str):
    systemQ = ("You are a mathematician, you are supposed to answer the given question. You need to output the answer "
    "in your final sentence like \"Therefore, the answer is ...\". The answer can only be one of the following forms:\n"
    "1. a numerical value like 0.1, no symbol and no unit at all.\n" 
    "2. a list of number like [2, 3, 4].\n"
    "3. True/False.\n"
    "4. an option like (a), (b), (c), (d)")

    # Greedy Decoding
    return None

    client = OpenAI()

    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
    )
    print(completion.choices[0].message)



if __name__ == "__main__":
    
    # Read Question-Answer Dataset
    with open("./datasets/theoremqa_test.json", 'r') as file:
        test_set = json.load(file)

    print("Test QA set: {} enteries".format(len(test_set)))

    if args.model:
        run_TinyLlama(test_set)

        







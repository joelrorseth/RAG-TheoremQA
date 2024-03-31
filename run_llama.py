import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import json
from tqdm import tqdm
from datetime import datetime
from prompt import *




def run_TinyLlama(entry, prompt_method):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

    prompt = ""
    if prompt_method == "standard":
        prompt = get_standard_prompt(entry['Question'], entry['Answer_type'])
    elif prompt_method == "cot":
        prompt = get_cot_prompt(entry['Question'], entry['Answer_type'])
    elif prompt_method == "tot_propose":
        prompt = get_tot_propose_prompt()
    message = [
        # {"role": "system", "content": "You can answer mathematical questions by reasoning"},
        {"role": "user", "content": prompt},
    ]   
    output = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = pipe(output, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)


    result = output[0]["generated_text"]
    # Strip instructions and inputs
    result = result.split("<|assistant|>")[-1]
    # Extract prediction in a rather trivial way
    prediction = '\n'.join(result.split('\n')[-2:]) # Prediction is just the last line of output
    return result, prediction

    

def run_Llama2_7b(entry, prompt_method):
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")

    prompt = ""
    if prompt_method == "standard":
        prompt = get_standard_prompt(entry['Question'], entry['Answer_type'])
    elif prompt_method == "cot":
        prompt = get_cot_prompt(entry['Question'], entry['Answer_type'])
    message = [
        # {"role": "system", "content": "You can answer mathematical questions by reasoning"},
        {"role": "user", "content": prompt},
    ]   
    output = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    output = pipe(output, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)

    result = output[0]["generated_text"]
    # Strip instructions and inputs
    # result = result.split("<|assistant|>")[-1]
    # Extract prediction in a rather trivial way
    prediction = '\n'.join(result.split('\n')[-2:]) # Prediction is just the last line of output
    return result, prediction



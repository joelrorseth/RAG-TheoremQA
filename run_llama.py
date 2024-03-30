import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import json
from tqdm import tqdm
from datetime import datetime
from prompt import get_cot_prompt



def run_TinyLlama(test_set):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    time = datetime.now()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"outputs/tinyllama_{time_string}.jsonl"

    # Experienment over the entire test set
    with open(file_name, 'w') as file:
        for entry in tqdm(test_set):
            # entry: dict["Question", "Answer", "Answer_type", "id", "theorem", "subfield", "field"]

            pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
            message = [
                # {"role": "system", "content": "You can answer mathematical questions by reasoning"},
                {"role": "user", "content": get_cot_prompt(entry['Question'], entry['Answer_type'])},
            ]   
            output = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            output = pipe(output, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)

        
            result = output[0]["generated_text"]
            # Strip instructions and inputs
            result = result.split("<|assistant|>")[-1]
            # Extract prediction in a rather trivial way
            prediction = '\n'.join(result.split('\n')[-2:]) # Prediction is just the last line of output
            tmp = {
                'id': entry['id'],
                'question': entry['Question'],
                'prediction': prediction,
                'answer': entry['Answer'],
                'rationale': result,
                'answer_type': entry['Answer_type'],
            }


            json.dump(tmp, file)
            file.write('\n')

    

def run_Llama2_7b(test_set):
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    time = datetime.now()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"outputs/llama2_7b_{time_string}.jsonl"


    # Experienment over the entire test set
    with open(file_name, 'w') as file:
        for entry in tqdm(test_set):
            # entry: dict["Question", "Answer", "Answer_type", "id", "theorem", "subfield", "field"]

            pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
            message = [
                # {"role": "system", "content": "You can answer mathematical questions by reasoning"},
                {"role": "user", "content": get_cot_prompt(entry['Question'], entry['Answer_type'])},
            ]   
            output = pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            output = pipe(output, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)

            result = output[0]["generated_text"]
            # Strip instructions and inputs
            # result = result.split("<|assistant|>")[-1]
            # Extract prediction in a rather trivial way
            prediction = '\n'.join(result.split('\n')[-2:]) # Prediction is just the last line of output
            tmp = {
                'id': entry['id'],
                'question': entry['Question'],
                'prediction': prediction,
                'answer': entry['Answer'],
                'rationale': result,
                'answer_type': entry['Answer_type'],
            }


            json.dump(tmp, file)
            file.write('\n')





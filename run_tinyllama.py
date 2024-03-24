import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from tqdm import tqdm
from datetime import datetime


def generate_data(test_set):
    for example in test_set:
        question = example['Question']

        if example['Answer_type']  == 'bool':
            instruction = "Please read a math problem, and then think step by step to derive the answer. The answer needs to be True or False."
        elif example['Answer_type']  == 'option':
            instruction = "Please read a math problem, and then think step by step to derive the answer. The answer needs to be (a), (b), (c) or (d)."
        else:
            instruction = "Please read a math problem, and then think step by step to derive the answer. The answer needs to be in numerical form."

        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:\n"

        del example['Picture']
        example['prompt'] = prompt
        yield example
    

def run_TinyLlama(test_set):
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    batch_size = 4
    
    dataset = Dataset.from_generator(lambda: generate_data(test_set))
    dataloader = DataLoader(dataset, batch_size=batch_size)


    time = datetime.now()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"outputs/tinyllama_{time_string}.jsonl"

    answer_mapping = {}
    for entry in test_set:
        answer_mapping[entry['id']] = entry['Answer']


    # Experienment over the entire test set
    with open(file_name, 'w') as file:
        for entry in tqdm(dataloader):

            pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.bfloat16, device_map="auto")
        
            prompts = []
            for prompt_str in entry["prompt"]:
                message = [
                    # {"role": "system", "content": "You can answer mathematical questions by reasoning"},
                    {"role": "user", "content": prompt_str},
                ]   
                prompts.append(pipe.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            
            outputs = pipe(prompts, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)

            for i, output in enumerate(outputs):
                result = output[0]["generated_text"]

                # Strip instructions and inputs
                result = result.split("<|assistant|>")[-1]

                # Extract prediction in a rather trivial way
                prediction = '\n'.join(result.split('\n')[-2:]) # Prediction is just the last line of output

                # prediction = None
                # for sent in result.split('\n')[::-1]:
                    # if entry['Answer_type'] == 'bool':
                    #     if len(sent) < 2:
                    #         continue
                    #     else:
                    #         if 'true' in sent.lower() or 'correct' in sent.lower():
                    #             prediction = 'True'
                    #         elif 'false' in sent.lower() or 'wrong' in sent.lower():
                    #             prediction = 'False'
                    #         elif ' not ' in sent.lower() or "n't " in sent.lower():
                    #             prediction = 'False'
                    #         else:
                    #             prediction = 'True'
                    #         break
                    # elif entry['Answer_type'] == 'option':
                    #     if '(a)' in sent.lower() or '(b)' in sent.lower() or '(c)' in sent.lower() or '(d)' in sent.lower():
                    #         prediction = sent
                    #         break
                    # else:
                    #     if ' is ' in sent.lower() or ' be ' in sent.lower() or ' are ' in sent.lower() or ' is: ' in sent.lower():
                    #         prediction = re.split(' be | is | are | is: ', sent)[-1].strip('.')
                    #         break
                    #     elif re.search('[0-9]', sent):
                    #         prediction = sent
                    #         break
                    #     else:
                    #         continue

                tmp = {
                    'id': entry['id'][i],
                    'question': entry['Question'][i],
                    'prediction': prediction,
                    'answer': answer_mapping[entry['id'][i]],
                    'rationale': result,
                    'answer_type': entry['Answer_type'][i],
                }

                json.dump(tmp, file)
                file.write('\n')






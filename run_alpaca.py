import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from datetime import datetime
import re


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

        del example['Answer']
        del example['Picture']
        example['prompt'] = prompt
        yield example
    

def run_alpaca(test_set):
    device = 'cpu'
    model_name = "chavinlo/alpaca-13b"
    batch_size = 4
    
    dataset = Dataset.from_generator(lambda: generate_data(test_set))
    dataloader = DataLoader(dataset, batch_size=batch_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left" # Why padding on the left?
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", offload_folder="offload",
                                                  torch_dtype=torch.float16)

    time = datetime.now()
    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")

    correct, wrong = 0, 0
    file_name = f"outputs/alpaca_{time_string}.jsonl"

    answer_mapping = {}
    for entry in test_set:
        answer_mapping[entry['id']] = entry['Answer']


    # Experienment over the entire test set
    with open(file_name, 'w') as file:
        for entry in tqdm(dataloader):
            batch = tokenizer(entry['prompt'], return_tensors='pt', add_special_tokens=False, padding=True)
            batch = {k : v.to(device) for k, v in batch.items()}


            outputs = model.generate(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                pad_token_id = tokenizer.eos_token_id,
                do_sample = False,
                max_new_tokens = 1024,
            )

            # for i, sequence in enumerate(outputs):
            #     sequence = sequence[batch['input_ids'].shape[1]:]
            #     result = tokenizer.decode(sequence, skip_special_tokens=True).strip()
            #     result = result.replace('</s><s>', '')
            #     prediction = None
            #     for sent in result.split('\n')[::-1]:
            #         if entry['Answer_type'] == 'bool':
            #             if len(sent) < 2:
            #                 continue
            #             else:
            #                 if 'true' in sent.lower() or 'correct' in sent.lower():
            #                     prediction = 'True'
            #                 elif 'false' in sent.lower() or 'wrong' in sent.lower():
            #                     prediction = 'False'
            #                 elif ' not ' in sent.lower() or "n't " in sent.lower():
            #                     prediction = 'False'
            #                 else:
            #                     prediction = 'True'
            #                 break
            #         elif entry['Answer_type'] == 'option':
            #             if '(a)' in sent.lower() or '(b)' in sent.lower() or '(c)' in sent.lower() or '(d)' in sent.lower():
            #                 prediction = sent
            #                 break
            #         else:
            #             if ' is ' in sent.lower() or ' be ' in sent.lower() or ' are ' in sent.lower() or ' is: ' in sent.lower():
            #                 prediction = re.split(' be | is | are | is: ', sent)[-1].strip('.')
            #                 break
            #             elif re.search('[0-9]', sent):
            #                 prediction = sent
            #                 break
            #             else:
            #                 continue

            #     if prediction is None:
            #         print(result)

            #     tmp = {
            #         'id': entry['id'][i],
            #         'question': entry['Question'][i],
            #         'prediction': prediction,
            #         'answer': answer_mapping[entry['id'][i]],
            #         'rationale': result,
            #         'answer_type': entry['Answer_type'][i],
            #         }

            #     json.dump(tmp, file)
            #     file.write('\n')





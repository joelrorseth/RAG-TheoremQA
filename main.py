import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from config import EVAL_DATA_PATH
from run_gpt import *
from run_llama import *
from src.prompting.cot import build_cot_prompt
from run_tot import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str) 
        # tinyllama       - 1.1B model
        # llama2-7b       - 7B meta model
        # gpt-3.5-turbo 
        # gpt-4
    
    # parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--all_examples", action='store_true') # If not, then exclude the Physics, Finance, CS questions
        # There are 442 math examples, 800 examples in total 
    parser.add_argument("--start_index", default=0, type=int)  # start index of examples to use
    parser.add_argument("--end_index", default=1, type=int)
    # parser.add_argument("--end_index", default=-1, type=int) # -1 indicate no bound for end index
    parser.add_argument("--prompt", default='tot', type=str)
        # standard - standard prompting
        # cot - chain of thought (with "Let's think step by step")
        # tot - Tree of thought
    parser.add_argument("--to_print", action='store_true', default=True)        # print tot log 

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Read Question-Answer Dataset
    with open(EVAL_DATA_PATH / "theoremqa/test.json", 'r') as file:
        test_set = json.load(file)


    if args.model not in ["tinyllama", "llama2-7b", "gpt-3.5-turbo", "gpt-4"]:
        raise NotImplementedError("The model is not indicated or not supported.")
    if args.model in ["gpt-3.5-turbo", "gpt-4"]:
        gpt_setup()


    if not args.all_examples:
        test_set = [example for example in test_set if example['field'] == "Math"]
        print("Only math examples are used.")
    # Remove not useful information in the examples
    for example in test_set:
        del example["Picture"]
        del example["source"]
        del example["explanation"]


    if args.end_index == -1:
        args.end_index = len(test_set)
    if args.start_index > args.end_index or args.end_index > len(test_set):
        raise IndexError("Invalid start or end index provided.")
    test_set = test_set[args.start_index: args.end_index]
    print("Test QA set contains {} enteries".format(len(test_set)))


    # Experiment over the test set:
    

    time = datetime.now()
    time_string = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"outputs/{args.model}_{time_string}.jsonl"

    # Experienment over the entire test set
    with open(file_name, 'w') as file:
        for entry in tqdm(test_set):
            # entry: dict["Question", "Answer", "Answer_type", "id", "theorem", "subfield", "field"]

            logs, result = [], None
            if args.model == "tinyllama":
                result, prediction = run_TinyLlama(entry, args.prompt)
            elif args.model == "llama2-7b":
                result, prediction = run_Llama2_7b(entry, args.prompt)
            elif args.model == "gpt-3.5-turbo" and args.prompt == "tot":
                step_limit = 3             # The number of steps for the evaluation tree
                breadth_limit = 1          # The number of new nodes to select at each step
                prediction, logs = run_tot(
                    entry, step_limit, breadth_limit, 
                    generate_method="propose", 
                    to_print=args.to_print,
                )
            elif args.model == "gpt-4":
                # result, prediction = run_gpt4(entry, args.prompt)
                pass


            tmp = {
                'id': entry['id'],
                'question': entry['Question'],
                'prediction': prediction,
                'answer': entry['Answer'],
                'rationale': result,
                'tot_logs': logs,
                'answer_type': entry['Answer_type'],
            }
            json.dump(tmp, file)
            file.write('\n')









    

    
    
        







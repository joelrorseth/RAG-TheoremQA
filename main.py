import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from run_gpt import run_gpt3_5_turbo, run_gpt4
from run_llama import run_TinyLlama, run_Llama2_7b



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyllama", type=str) 
        # tinyllama       - 1.1B model
        # llama2-7b       - 7B meta model
        # gpt-3.5-turbo 
        # gpt-4
    
    # parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--all_examples", action='store_true') # If not, then exclude the Physics, Finance, CS questions
        # There are 442 math examples, 800 examples in total 
    parser.add_argument("--start_index", default=0, type=int)  # start index of examples to use
    parser.add_argument("--end_index", default=-1, type=int) # -1 indicate no bound for end index
    parser.add_argument("--dryrun", action='store_true')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    # Read Question-Answer Dataset
    with open("./datasets/theoremqa_test.json", 'r') as file:
        test_set = json.load(file)


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
    

    if args.model == "tinyllama":
        run_TinyLlama(test_set)
    elif args.model == "llama2-7b":
        run_Llama2_7b(test_set)
    elif args.model == "gpt-3.5-turbo":
        # run_gpt3_5_turbo(test_set)
        pass
    elif args.model == "gpt-4":
        # run_gpt4(test_set)
        pass
    else:
        raise NotImplementedError("The model is not indicated or not supported.")

        







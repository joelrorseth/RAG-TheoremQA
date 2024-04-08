import re
from run_gpt import run_tot_gpt
from src.prompting.tot import build_tot_propose_prompt, build_tot_sample_prompt, build_tot_vote_prompt


# Given gpt output with votes, take out the vote result
def unwrap_vote(outputs: list, n_candidate: int) -> list:
    votes = [0] * n_candidate
    for output in outputs:
        pattern = r".*best choice is .*(\d+).*"
        match = re.match(pattern, output, re.DOTALL)
        if match:
            vote = int(match.groups()[0])
            if vote in range(n_candidate):
                votes[vote] += 1
        else:
            print(f'vote has no match: {[output]}')
    return votes


def run_tot(entry, step_limit: int, breadth_limit: int, generate_method="sample", to_print=False):
    # Running Tree of Thought on gpt-3.5-turbo model
    # 
    # step_limit defines the maximum number of thought steps to decompose
    # breadth limit defines the number of states to search for at each steps
    # generate_method : "sample"   - iid sample from a cot prompt
    #                   "propose"  - propose different prompts in a limited search space
    #
    # Return: nodes : final set of nodes in the tree
    #         logs  : log that details the nodes and selected nodes at each steps

    nodes = [""]
    logs = []
    question, answer_type = entry['Question'], entry['Answer_type']
    for step in range(step_limit):
             
        new_nodes = []
        if generate_method == "sample":
            n_sample = 2                  # The max number of samples we want to get
            stop = '\n'                      # Stop with new line symbol, unless it is the last step of reasoning.
            if step == step_limit - 1:
                stop = None 
            for node in nodes:
                sample_prompt = build_tot_sample_prompt(question, node, answer_type)
                samples = run_tot_gpt("gpt-3.5-turbo", sample_prompt, n_sample=n_sample, stop=stop)
                # new nodes are original nodes augmented with new samples
                new_nodes.extend([node + sample + '\n' for sample in samples])
        elif generate_method == 'propose':
            n_propose = 2
            if step == step_limit - 1:
                # Sample the answer if this is the last step, this helps the final answer to be in an indicated format
                for node in nodes:
                    sample_prompt = build_tot_sample_prompt(question, node, answer_type)
                    samples = run_tot_gpt("gpt-3.5-turbo", sample_prompt, n_sample=n_propose, stop=None)
                    new_nodes.extend([node + sample + '\n' for sample in samples])
            else:
                for node in nodes:
                    proposal_prompt = build_tot_propose_prompt(question, node, n_propose)
                    proposals = run_tot_gpt("gpt-3.5-turbo", proposal_prompt, n_sample=1, stop=None)[0].split('\n')
                    new_nodes.extend([node + p + '\n' for p in proposals])

        # Evaluate the generated outputs, here we only consider the method of "vote across states".
        # Then greedy selecting the first {breadth_limit} nodes to retain
        n_evaluate = 3    # Number of votes 
        vote_prompt = build_tot_vote_prompt(entry['Question'], new_nodes)
        vote_outputs = run_tot_gpt("gpt-3.5-turbo", vote_prompt, n_sample=n_evaluate, stop=None)
        votes = unwrap_vote(vote_outputs, len(new_nodes))

        ids = list(range(len(new_nodes)))
        select_ids = sorted(ids, key=lambda x: votes[x], reverse=True)[:breadth_limit]
        selected_new_nodes = [new_nodes[id] for id in select_ids]

        if to_print:
            print(f"== Step: {step} ==")
            for node, vote in sorted(zip(new_nodes, votes), key=lambda x: x[1], reverse=True):
                print(f"Node: {node}, Vote: {vote}\n")
        
        logs.append({'step': step, 'question': entry['Question'], 
                     'nodes': nodes, 
                     'new_nodes': new_nodes, 
                     'values': votes, 
                     'selected_new_nodes': selected_new_nodes})
        nodes = selected_new_nodes

    
    # Extract prediction and answer from the outcome
    best_node = nodes[0]
    prediction = '\n'.join(best_node.split('\n')[-2:]) # Prediction is just the last two lines of output
    return prediction, logs
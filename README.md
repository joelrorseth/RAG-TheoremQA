# Theorem QA with RAG and Prompting

In this project, we design several LLM prompting
strategies in an attempt to beat state-of-the-art
performance on TheoremQA questions. We implement
several prompting strategies that employ
recent prompt engineering techniques, including
Chain-of-Thought with self-consistency,
Tree of Thoughts, and retrieval-augmented
generation.

## Installation

Using Python 3.10:
```
python -m venv venv
source venv/bin/activate
pip install requirements.txt
```

## Running Experiments

After installation, you can run the following
file to build all indexes and execute all
experiments. We have added the predictions
and results files from our own runs under
`data/predictions` and `data/results`, so
running this script will overwrite these files.
You'll also need to provide your OpenAI API key
like so:

```
export OPENAI_API_KEY=<YOUR KEY HERE>
python run_experiments.py
```

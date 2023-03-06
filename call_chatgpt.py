import json
import os.path

import openai
import tqdm
import argparse
from data import DATA_DIR


def main(args):
    # Provide OpenAI API key
    api_key = input("Please provide an OpenAI API key:\n")
    openai.api_key = api_key

    dataset = []
    with open(os.path.join(DATA_DIR, f'{args.dataset_name}.jsonl')) as in_file:
        for line in in_file:
            dataset.append(json.loads(line))

    for idx, example in tqdm.tqdm(enumerate(dataset)):
        if args.model_name == 'gpt-3.5-turbo':
            try:
                response = openai.ChatCompletion.create(
                  model=args.model_name,
                  messages=[
                        {"role": "user", "content": example['input_text']},
                    ],
                  max_tokens=100
                )
                dataset[idx]['prediction'] = response['choices'][0]['message']['content']
            except:
                dataset[idx]['prediction'] = None
        else:
            try:
                response = openai.Completion.create(
                  model="gpt-3.5",
                  prompt=example['input_text'] + " The answer is: "
                )
                dataset[idx]['prediction'] = response['choices'][0]['message']['content']
            except:
                dataset[idx]['prediction'] = None

    with open(os.path.join(DATA_DIR, f'{args.dataset_name}_{args.model_name}_predictions.jsonl'), 'w') as file:
        for example in dataset:
            file.write(json.dumps(example) + '\n')


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--dataset_name", type=str, default='ledgar', help="Name of dataset as stored on HF")
parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', help="GPT model name")
args = parser.parse_args()

main(args)

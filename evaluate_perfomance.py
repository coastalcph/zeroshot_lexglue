import json
import numpy as np
import os
from sklearn.metrics import classification_report
from datasets import load_dataset
from data import DATA_DIR
import argparse

def main(args):
    predict_dataset = load_dataset("lexlms/lex_glue_v2", args.dataset_name, split="test",
                                   use_auth_token='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')

    if args.multi_label:
        label_names = [f'{label_name}'.lower() for idx, label_name in enumerate(predict_dataset.features['labels'].feature.names)]
        label_names.append(f'None'.lower())
    else:
        label_names = [f'{label_name}'.lower() for idx, label_name in enumerate(predict_dataset.features['label'].names)]

    dataset = []
    with open(os.path.join(DATA_DIR, f'{args.dataset_name}_{args.model_name}_predictions.jsonl'), 'w') as file:
        for line in file:
            dataset.append(json.loads(line))

    labels = np.zeros((len(dataset), len(label_names)))
    predictions = np.zeros((len(dataset), len(label_names)))
    nones = 0
    for idx, example in enumerate(dataset):
        if example['prediction'] is not None:
            for l_idx, label_name in enumerate(label_names):
                if label_name in dataset[idx]['answer'].lower():
                    labels[idx][l_idx] = 1
                if label_name in example['prediction'].lower():
                    predictions[idx][l_idx] = 1
        else:
            nones += 1

    print(f'{nones} question unanswered!\n')
    print(classification_report(y_true=labels, y_pred=predictions, target_names=label_names, zero_division=0))


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--dataset_name", type=str, default='unfair_tos', help="Name of dataset as stored on HF")
parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', help="GPT model name")
parser.add_argument("--multi_label", type=bool, default='gpt-3.5-turbo', help="GPT model name")

args = parser.parse_args()

main(args)


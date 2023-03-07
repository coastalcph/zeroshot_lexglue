import json
import numpy as np
import os
from sklearn.metrics import classification_report
from datasets import load_dataset
from data import DATA_DIR
import argparse
import random


def main(args):
    # Load test dataset and labels
    predict_dataset = load_dataset("lexlms/lex_glue_v2", args.dataset_name, split="test",
                                   use_auth_token='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')
    if args.multi_label:
        label_names = [f'{label_name}'.lower() for idx, label_name in enumerate(predict_dataset.features['labels'].feature.names)] + ['none']
    elif args.dataset_name == 'case_hold':
        label_names = [f'Choice {idx}' for idx in range(5)]
    else:
        label_names = [f'{label_name}'.lower() for idx, label_name in
                       enumerate(predict_dataset.features['label'].names)]
    random.seed(42)
    random_ids = random.sample(range(len(predict_dataset)), k=1000)
    predict_dataset = predict_dataset.select(random_ids)

    labels = np.zeros((len(predict_dataset), len(label_names)))
    predictions = np.zeros((len(predict_dataset), len(label_names)))
    for idx, example in enumerate(predict_dataset):
        for l_idx, label_name in enumerate(label_names):
            if args.multi_label and l_idx in example['labels']:
                labels[idx][l_idx] = 1
            elif not args.multi_label and l_idx == example['label']:
                labels[idx][l_idx] = 1
        if args.multi_label:
            random_ids = random.sample(range(len(label_names)), 5)
            for random_idx in random_ids:
                predictions[idx][random_idx] = 1
        else:
            random_idx = random.choice(range(len(label_names)))
            predictions[idx][random_idx] = 1

    print(classification_report(y_true=labels, y_pred=predictions, target_names=label_names, zero_division=0, digits=3))


parser = argparse.ArgumentParser(description='Prompting GPT')
parser.add_argument("--dataset_name", type=str, default='case_hold', help="Name of dataset as stored on HF")
parser.add_argument("--multi_label", type=bool, default=False, help="GPT model name")

args = parser.parse_args()

main(args)

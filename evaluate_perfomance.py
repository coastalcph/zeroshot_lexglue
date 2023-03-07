import json
import numpy as np
import os
from sklearn.metrics import classification_report
from datasets import load_dataset
from data import DATA_DIR
import argparse

EUROVOC_CONCEPTS = ['political framework', 'politics and public safety', 'executive power and public service',
                    'international affairs', 'cooperation policy', 'international security', 'defence',
                    'EU institutions and European civil service', 'European Union law', 'European construction',
                    'EU finance', 'civil law', 'criminal law', 'international law', 'rights and freedoms',
                    'economic policy',
                    'economic conditions', 'regions and regional policy', 'national accounts', 'economic analysis',
                    'trade policy', 'tariff policy', 'trade', 'international trade', 'consumption', 'marketing',
                    'distributive trades', 'monetary relations', 'monetary economics',
                    'financial institutions and credit',
                    'free movement of capital', 'financing and investment', 'public finance and budget policy',
                    'budget',
                    'taxation', 'prices', 'social affairs', 'social protection', 'health', 'documentation',
                    'communications',
                    'information and information processing', 'information technology and data processing',
                    'natural and applied sciences', 'business organisation', 'business classification', 'management',
                    'accounting', 'competition', 'employment', 'labour market',
                    'organisation of work and working conditions', 'personnel management and staff remuneration',
                    'transport policy', 'organisation of transport', 'land transport',
                    'maritime and inland waterway transport', 'air and space transport', 'environmental policy',
                    'natural environment', 'deterioration of the environment', 'agricultural policy',
                    'agricultural structures and production', 'farming systems', 'cultivation of agricultural land',
                    'means of agricultural production', 'agricultural activity', 'fisheries', 'plant product',
                    'animal product', 'processed agricultural produce', 'beverages and sugar', 'foodstuff',
                    'agri-foodstuffs', 'food technology', 'production', 'technology and technical regulations',
                    'research and intellectual property', 'energy policy', 'coal and mining industries', 'oil industry',
                    'electrical and nuclear industries', 'industrial structures and policy', 'chemistry',
                    'iron, steel and other metal industries', 'mechanical engineering',
                    'electronics and electrical engineering', 'building and public works', 'wood industry',
                    'leather and textile industries', 'miscellaneous industries', 'Europe',
                    'regions of EU Member States',
                    'America', 'Africa', 'Asia and Oceania', 'economic geography', 'political geography',
                    'overseas countries and territories', 'United Nations']


def main(args):
    predict_dataset = load_dataset("lexlms/lex_glue_v2", args.dataset_name, split="test",
                                   use_auth_token='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')

    if args.multi_label:
        if args.dataset_name != 'eurlex':
            label_names = [f'{label_name}'.lower() for idx, label_name in
                           enumerate(predict_dataset.features['labels'].feature.names)] + ['none']
        else:
            label_names = EUROVOC_CONCEPTS
    else:
        label_names = [f'{label_name}'.lower() for idx, label_name in
                       enumerate(predict_dataset.features['label'].names)]

    dataset = []
    name_extension = f'_few_shot-{args.few_shot_k}' if args.few_shot_k else ''
    with open(os.path.join(DATA_DIR, f'{args.dataset_name}_{args.model_name}_predictions{name_extension}.jsonl')) as file:
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
    print(classification_report(y_true=labels, y_pred=predictions, target_names=label_names, zero_division=0, digits=3))


parser = argparse.ArgumentParser(description='Evaluate GPT')
parser.add_argument("--dataset_name", type=str, default='scotus', help="Name of dataset as stored on HF")
parser.add_argument("--model_name", type=str, default='gpt-3.5-turbo', help="GPT model name")
parser.add_argument("--multi_label", type=bool, default=False, help="GPT model name")
parser.add_argument("--few_shot_k", type=int, default=0, help="GPT model name")

args = parser.parse_args()

main(args)

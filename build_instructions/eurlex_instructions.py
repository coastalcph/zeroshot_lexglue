import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random

predict_dataset = load_dataset("lexlms/lex_glue_v2", 'eurlex', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['labels'].feature.names
label_names = ['political framework', 'politics and public safety', 'executive power and public service',
               'international affairs', 'cooperation policy', 'international security', 'defence',
               'EU institutions and European civil service', 'European Union law', 'European construction',
               'EU finance', 'civil law', 'criminal law', 'international law', 'rights and freedoms', 'economic policy',
               'economic conditions', 'regions and regional policy', 'national accounts', 'economic analysis',
               'trade policy', 'tariff policy', 'trade', 'international trade', 'consumption', 'marketing',
               'distributive trades', 'monetary relations', 'monetary economics', 'financial institutions and credit',
               'free movement of capital', 'financing and investment', 'public finance and budget policy', 'budget',
               'taxation', 'prices', 'social affairs', 'social protection', 'health', 'documentation', 'communications',
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
               'leather and textile industries', 'miscellaneous industries', 'Europe', 'regions of EU Member States',
               'America', 'Africa', 'Asia and Oceania', 'economic geography', 'political geography',
               'overseas countries and territories', 'United Nations']
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

with open(os.path.join(DATA_DIR, 'eurlex.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"]
        shortened_text = ' '.join(text.split(' ')[:512])
        text_input = f'Given the following excerpt from an EU law:\n"{text}"\n'
        text_input += 'Which topics are relevant out of the following options:\n'
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += 'The relevant options are:'
        print(text_input)
        answer = ", ".join([label_names[label] for idx, label in sorted(enumerate(sample['labels']))])
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'The right options are: {answer}')
        print('-' * 100)

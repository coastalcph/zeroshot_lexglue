import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
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


# Compute templated text tokens
templated_text = TEMPLATES['eurlex']['INPUT_INTRODUCTORY_TEXT'] + '\n" "\n'
templated_text += TEMPLATES['eurlex']['OPTIONS_PRESENTATION_TEXT']
for end_idx, label_name in enumerate(label_names):
    templated_text += f'- {label_name}\n'
templated_text += TEMPLATES['eurlex']['QUESTION_TEXT']

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
templated_text_length = len(tokenizer.encode(templated_text))

total_input = ''
with open(os.path.join(DATA_DIR, 'instruction-following-examples', 'eurlex.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"]
        words = text.split(' ')
        for threshold in [512, 450, 400]:
            shortened_text = ' '.join(text.split(' ')[:threshold])
            input_text_length = len(tokenizer.encode(shortened_text))
            if templated_text_length + input_text_length <= 4000:
                break
        text_input = TEMPLATES['eurlex']['INPUT_INTRODUCTORY_TEXT'] + f'"{shortened_text}"\n\n'
        text_input += TEMPLATES['eurlex']['OPTIONS_PRESENTATION_TEXT']
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += TEMPLATES['eurlex']['QUESTION_TEXT']
        print(text_input)
        answer = ", ".join([label_names[label] for idx, label in sorted(enumerate(sample['labels']))])
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f"{TEMPLATES['eurlex']['QUESTION_TEXT']} {answer}")
        print('-' * 100)
        total_input += text_input


# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




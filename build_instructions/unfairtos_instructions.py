import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random

predict_dataset = load_dataset("lexlms/lex_glue_v2", 'unfair_tos', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['labels'].feature.names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Prompt templated text
INPUT_INTRODUCTORY_TEXT = f'Given the following sentence from an online Term of Services:'
OPTIONS_PRESENTATION_TEXT = 'The sentence is unfair with respect to some of the following options:\n'
QUESTION_TEXT = 'The relevant options are:'

with open(os.path.join(DATA_DIR, 'unfair_tos.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"].replace(' ,', ',').replace(' .', '.').replace('\n', ' ').replace('`` ', '\'').replace(' \'\'', '\'').strip()
        text_input = INPUT_INTRODUCTORY_TEXT + f'"{text}"\n\n'
        text_input += OPTIONS_PRESENTATION_TEXT
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += f'- None\n'
        text_input += QUESTION_TEXT
        print(text_input)
        if len(sample['labels']):
            answer = ", ".join([label_names[label] for idx, label in sorted(enumerate(sample['labels']))])
        else:
            answer = f'None'
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'{QUESTION_TEXT} {answer}')
        print('-'*100)

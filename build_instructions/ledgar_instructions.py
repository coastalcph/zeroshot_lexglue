import json
import random
import os
from data import DATA_DIR
from datasets import load_dataset


predict_dataset = load_dataset("lexlms/lex_glue_v2", 'ledgar', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['label'].names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Prompt templated text
INPUT_INTRODUCTORY_TEXT = f'Given the following contractual section:'
OPTIONS_PRESENTATION_TEXT = 'There is an appropriate section title out of the following options:\n'
QUESTION_TEXT = 'The most appropriate option is:'

with open(os.path.join(DATA_DIR, 'ledgar.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"].replace(' ,', ',').replace(' .', '.').replace('\n', ' ').replace('`` ', '\'').replace(' \'\'', '\'').strip()
        text_input = INPUT_INTRODUCTORY_TEXT + f'\n"{text}"\n\n'
        text_input += OPTIONS_PRESENTATION_TEXT
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += QUESTION_TEXT
        print(text_input)
        answer = label_names[sample['label']]
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'{QUESTION_TEXT} {answer}')
        print('-'*100)

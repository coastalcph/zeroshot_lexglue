import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random

predict_dataset = load_dataset("lexlms/lex_glue_v2", 'case_hold', split="test",
                               use_auth_token='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')
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Prompt templated text
INPUT_INTRODUCTORY_TEXT = 'Given the following excerpt from a US opinion:'
OPTIONS_PRESENTATION_TEXT = 'The [Masked Holding] is a placeholder for one of the following options:\n'
QUESTION_TEXT = 'The relevant option is:'


with open(os.path.join(DATA_DIR, 'case_hold.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text_input = INPUT_INTRODUCTORY_TEXT + f'\n"{sample["contexts"][0].replace("<HOLDING>", "[Masked Holding]")}"\n\n'
        text_input += OPTIONS_PRESENTATION_TEXT
        for end_idx, ending in enumerate(sample['endings']):
            text_input += f'- {ending}\n'
        text_input += QUESTION_TEXT
        print(text_input)
        file.write(json.dumps({'input_text': text_input, 'answer': sample["endings"][sample["label"]]}) + '\n')
        print(f'{QUESTION_TEXT} {sample["endings"][sample["label"]]}')
        print('-'*100)
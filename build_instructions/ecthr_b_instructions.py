import json
import os
from data import DATA_DIR
from datasets import load_dataset

predict_dataset = load_dataset("lex_glue", 'ecthr_b', split="test")
label_names = predict_dataset.features['labels'].feature.names

with open(os.path.join(DATA_DIR, 'ecthr_b.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = '\n'.join(sample["text"])
        shortened_text = ' '.join(text.split(' ')[:4096])
        text_input = f'Given the following facts from a European Court of Human Rights (ECtHR) case:\n"{shortened_text}"\n\n'
        text_input += 'Which article(s) of ECHR are relevant out of the following options:\n'
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- Article {label_name}\n'
        text_input += f'- None\n'
        text_input += 'The relevant options are:'
        print(text_input)
        if len(sample['labels']):
            answer = ", ".join([f"Article {label_names[label]}" for label in sorted(sample['labels'])])
        else:
            answer = f'None'
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'The relevant options are: {answer}')
        print('-'*100)
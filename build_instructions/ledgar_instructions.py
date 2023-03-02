import json
import random
import os
from data import DATA_DIR
from datasets import load_dataset

predict_dataset = load_dataset("lex_glue", 'ledgar', split="test")
label_names = predict_dataset.features['label'].names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

with open(os.path.join(DATA_DIR, 'ledgar.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"].replace(' ,', ',').replace(' .', '.').replace('\n', ' ').replace('`` ', '\'').replace(' \'\'', '\'').strip()
        text_input = f'Given the following contractual section: "{text}"\n'
        text_input += 'What is the most appropriate section title out of the following options:\n'
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += 'The most appropriate options is:'
        print(text_input)
        answer = label_names[sample['label']]
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'The best option is: {answer}')
        print('-'*100)
import json
import os
from data import DATA_DIR
from datasets import load_dataset

predict_dataset = load_dataset("lex_glue", 'scotus', split="test")
# label_names = predict_dataset.features['label'].names
label_names = ['Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process', 'Privacy', 'Attorneys', 
                'Unions', 'Economic Activity', 'Judicial Power', 'Federalism', 'Interstate Relations',
                'Federal Taxation', 'Miscellaneous']
with open(os.path.join(DATA_DIR, 'scotus.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"]
        shortened_text = ' '.join(text.split(' ')[:4096])
        text_input = f'Given the following opinion from the Supreme Court of USA (SCOTUS):\n"{shortened_text}"\n\n'
        text_input += 'Which topics are relevant out of the following options:\n'
        for end_idx, label_name in enumerate(label_names):
            text_input += f'-  {label_name}\n'
        text_input += 'The correct option is:'
        print(text_input)
        answer = label_names[sample['label']]
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f'The relevant option is: {answer}')
        print('-'*100)
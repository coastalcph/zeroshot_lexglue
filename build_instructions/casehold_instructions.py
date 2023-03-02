import json
import os
from data import DATA_DIR
from datasets import load_dataset

predict_dataset = load_dataset("lex_glue", 'case_hold', split="test")

with open(os.path.join(DATA_DIR, 'case_hold.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text_input = f'Given the following excerpt from a US opinion:\n"{sample["contexts"][0].replace("<HOLDING>", "[Masked Holding]")}"\n\n'
        text_input += 'The [Masked Holding] is a placeholder for one of the following options:\n'
        for end_idx, ending in enumerate(sample['endings']):
            text_input += f'- {ending}\n'
        text_input += 'The relevant option is:'
        print(text_input)
        file.write(json.dumps({'input_text': text_input, 'answer': sample["endings"][sample["label"]]}) + '\n')
        print(f'The right options is: {sample["endings"][sample["label"]]}')
        print('-'*100)
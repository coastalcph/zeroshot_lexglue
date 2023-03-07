import json
import random
import os
from data import DATA_DIR
from datasets import load_dataset
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
predict_dataset = load_dataset("lexlms/lex_glue_v2", 'ledgar', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['label'].names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

total_input = ''
with open(os.path.join(DATA_DIR, 'ledgar.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"].replace(' ,', ',').replace(' .', '.').replace('\n', ' ').replace('`` ', '\'').replace(' \'\'', '\'').strip()
        text_input = TEMPLATES['ledgar']['INPUT_INTRODUCTORY_TEXT'] + f'"{text}"\n\n'
        text_input += TEMPLATES['ledgar']['OPTIONS_PRESENTATION_TEXT']
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += TEMPLATES['ledgar']['QUESTION_TEXT']
        print(text_input)
        answer = label_names[sample['label']]
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f"{TEMPLATES['ledgar']['QUESTION_TEXT']} {answer}")
        print('-'*100)
        total_input += text_input

# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




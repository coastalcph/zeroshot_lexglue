import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
predict_dataset = load_dataset("lexlms/lex_glue_v2", 'ecthr_b', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['labels'].feature.names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Used in ecthr_b3 template
# label_names = ['Article 2 - Right to life', 'Article 3 - Prohibition of torture',  'Article 5 - Right to liberty and security',
#                'Article 6 - Right to a fair trial',  'Article 8 - Right to respect for private and family life',
#                'Article 9 - Freedom of thought, conscience and religion', 'Article 10 - Freedom of expression',
#                'Article 11 - Freedom of assembly and association', 'Article 14 - Prohibition of discriminatioN',
#                'Article P1-1 - Protection of property']
# Compute templated text tokens
templated_text = TEMPLATES['ecthr_b']['INPUT_INTRODUCTORY_TEXT'] + '\n" "\n'
templated_text += TEMPLATES['ecthr_b']['OPTIONS_PRESENTATION_TEXT']
for end_idx, label_name in enumerate(label_names):
    templated_text += f'- {label_name}\n'
templated_text += f'- None of the above\n'
templated_text += TEMPLATES['ecthr_b']['QUESTION_TEXT']

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
templated_text_length = len(tokenizer.encode(templated_text))

total_input = ''
with open(os.path.join(DATA_DIR, 'instruction-following-examples', 'ecthr_b3.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = '\n'.join(sample["text"])
        words = text.split(' ')
        for threshold in [4000, 3800, 3600, 3400, 3200, 3000]:
            shortened_text = ' '.join(text.split(' ')[:threshold])
            input_text_length = len(tokenizer.encode(shortened_text))
            if templated_text_length + input_text_length <= 4000:
                break
        text_input = TEMPLATES['ecthr_b']['INPUT_INTRODUCTORY_TEXT'] + f'\n"{shortened_text}"\n\n'
        text_input += TEMPLATES['ecthr_b']['OPTIONS_PRESENTATION_TEXT']
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += f'- None of the above\n'
        text_input += TEMPLATES['ecthr_b']['QUESTION_TEXT']
        print(text_input)
        if len(sample['labels']):
            answer = ", ".join([f"{label_names[label]}" for label in sorted(sample['labels'])])
        else:
            answer = f'None'
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f"{TEMPLATES['ecthr_b']['QUESTION_TEXT']} {answer}")
        print('-'*100)
        total_input += text_input


# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




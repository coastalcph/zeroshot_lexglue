import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
predict_dataset = load_dataset("lexlms/lex_glue_v2", 'scotus', split="test",
                               use_auth_token='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')
label_names = ['Criminal Procedure', 'Civil Rights', 'First Amendment', 'Due Process', 'Privacy', 'Attorneys', 
                'Unions', 'Economic Activity', 'Judicial Power', 'Federalism', 'Interstate Relations',
                'Federal Taxation', 'Miscellaneous']
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Compute templated text tokens
templated_text = TEMPLATES['scotus']['INPUT_INTRODUCTORY_TEXT'] + '\n" "\n\n'
templated_text += TEMPLATES['scotus']['OPTIONS_PRESENTATION_TEXT']
for end_idx, label_name in enumerate(label_names):
    templated_text += f'- {label_name}\n'
templated_text += TEMPLATES['scotus']['QUESTION_TEXT']

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
templated_text_length = len(tokenizer.encode(templated_text))

total_input = ''
with open(os.path.join(DATA_DIR, 'scotus.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"]
        words = text.split(' ')
        for threshold in [4000, 3800, 3600, 3400, 3200, 3000, 2800, 2600, 2400, 2000]:
            shortened_text = ' '.join(text.split(' ')[:threshold])
            input_text_length = len(tokenizer.encode(shortened_text))
            if templated_text_length + input_text_length <= 4000:
                break
        text_input = TEMPLATES['scotus']['INPUT_INTRODUCTORY_TEXT'] + f'\n"{shortened_text}"\n\n'
        text_input += 'Which topics are relevant out of the following options:\n'
        for end_idx, label_name in enumerate(label_names):
            text_input += f'-  {label_name}\n'
        text_input += TEMPLATES['scotus']['QUESTION_TEXT']
        print(text_input)
        answer = label_names[sample['label']]
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f"{TEMPLATES['scotus']['QUESTION_TEXT']} {answer}")
        print('-'*100)
        total_input += text_input

# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




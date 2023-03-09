import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
predict_dataset = load_dataset("lexlms/lex_glue_v2", 'unfair_tos', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['labels'].feature.names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

total_input = ''
with open(os.path.join(DATA_DIR, 'instruction-following-examples', 'unfair_tos.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = sample["text"].replace(' ,', ',').replace(' .', '.').replace('\n', ' ').replace('`` ', '\'').replace(' \'\'', '\'').strip()
        text_input = TEMPLATES['unfair_tos']['INPUT_INTRODUCTORY_TEXT'] + f'"{text}"\n\n'
        text_input += TEMPLATES['unfair_tos']['OPTIONS_PRESENTATION_TEXT']
        for end_idx, label_name in enumerate(label_names):
            text_input += f'- {label_name}\n'
        text_input += f'- None\n'
        text_input += TEMPLATES['unfair_tos']['QUESTION_TEXT']
        print(text_input)
        if len(sample['labels']):
            answer = ", ".join([label_names[label] for idx, label in sorted(enumerate(sample['labels']))])
        else:
            answer = f'None'
        file.write(json.dumps({'input_text': text_input, 'answer': answer}) + '\n')
        print(f"{TEMPLATES['unfair_tos']['QUESTION_TEXT']} {answer}")
        print('-'*100)
        total_input += text_input

# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




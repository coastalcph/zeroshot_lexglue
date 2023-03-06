import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken

predict_dataset = load_dataset("lexlms/lex_glue_v2", 'ecthr_b', split="test",
                               use_auth_token='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')
label_names = predict_dataset.features['labels'].feature.names
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

# Prompt templated text
INPUT_INTRODUCTORY_TEXT = f'Given the following facts from a European Court of Human Rights (ECtHR) case:'
OPTIONS_PRESENTATION_TEXT = 'Which article(s) of ECHR are relevant, if any, out of the following options:\n'
QUESTION_TEXT = 'The relevant options are:'

templated_text = INPUT_INTRODUCTORY_TEXT + '\n" "\n'
templated_text += OPTIONS_PRESENTATION_TEXT
for end_idx, label_name in enumerate(label_names):
    templated_text += f'- {label_name}\n'
templated_text += QUESTION_TEXT


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
templated_text_length = len(tokenizer.encode(templated_text))

with open(os.path.join(DATA_DIR, 'ecthr_b.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text = '\n'.join(sample["text"])
        words = text.split(' ')
        for threshold in [4000, 3800, 3600, 3400, 3200, 3000]:
            shortened_text = ' '.join(text.split(' ')[:threshold])
            input_text_length = len(tokenizer.encode(shortened_text))
            if templated_text_length + input_text_length <= 4000:
                break
        text_input = INPUT_INTRODUCTORY_TEXT + f'\n"{shortened_text}"\n\n'
        text_input += OPTIONS_PRESENTATION_TEXT
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
        print(f'{QUESTION_TEXT} {answer}')
        print('-'*100)

import json
import os
from data import DATA_DIR
from datasets import load_dataset
import random
import tiktoken
from templates import TEMPLATES

# Load test dataset and labels
predict_dataset = load_dataset("lexlms/lex_glue_v2", 'case_hold', split="test",
                               use_auth_token='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')
random.seed(42)
random_ids = random.sample(range(len(predict_dataset)), k=1000)
predict_dataset = predict_dataset.select(random_ids)

total_input = ''
with open(os.path.join(DATA_DIR, 'instruction-following-examples', 'case_hold.jsonl'), 'w') as file:
    for idx, sample in enumerate(predict_dataset):
        text_input = TEMPLATES['case_hold']['INPUT_INTRODUCTORY_TEXT'] + f'\n"{sample["contexts"][0].replace("<HOLDING>", "[Masked Holding]")}"\n\n'
        text_input += TEMPLATES['case_hold']['OPTIONS_PRESENTATION_TEXT']
        for end_idx, ending in enumerate(sample['endings']):
            text_input += f'- {ending}\n'
        text_input += TEMPLATES['case_hold']['QUESTION_TEXT']
        print(text_input)
        file.write(json.dumps({'input_text': text_input, 'choices': sample["endings"], 'answer': sample["endings"][sample["label"]]}) + '\n')
        print(f"{TEMPLATES['case_hold']['QUESTION_TEXT']} {sample['endings'][sample['label']]}")
        print('-'*100)
        total_input += text_input

# Count tokens and cost
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
total_n_tokens = len(tokenizer.encode(total_input)) + 100 * 1000
print(f'The total number of tokens is {total_n_tokens}, with an '
      f'estimated processing cost of {total_n_tokens * (0.002/1000):.2f}$.')




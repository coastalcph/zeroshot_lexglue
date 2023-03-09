# Zero-shot Evaluation with OpenAI GPT-3.5 Models on the LexGLUE benchmark

This project is related to the pre-print paper: ["ChatGPT may Pass the Bar Exam soon, but has a Long Way to Go for the LexGLUE benchmark"](http://) by Chalkidis (2023). 

Following the work ["GPT Takes the Bar Exam"](https://arxiv.org/abs/2212.14402) of Bommarito II and Katz (2023), we evaluate the latest OpenAI's GPT-3.5 model `gpt-3.5-turbo`, (v. March 2023), the first available ChatGPT, on legal text classification tasks from the LexGLUE benchmark in a zero-shot fashion providing examples in a templated instruction-following format, similar to those used by  ["Scaling Instruction-Finetuned Language Models"](https://arxiv.org/abs/2210.11416) (Chung et al., 2022)..  We find that ChatGPT achieves an average micro-F1 score of 49.0% across LexGLUE tasks in a zero-shot setting, significantly in excess of the baseline guessing rates, while the model performs exceptionally well in some datasets achieving micro-F1 scores of 62.8% and 70.1% in the ECtHR B and LEDGAR datasets.

If you mention or build on top of this work, please cite:
```
@article{chalkidis-2023-lexglue-chatgpt,
  title={ChatGPT may pass the Legal Bar Exams soon, but sucks in LexGLUE},
  author={Ilias Chalkidis},
  journal={arXiv:xxxx.xxxx},
  year={2023},
}
```

# How things work?

We evaluate GPT-3.5 models in a zero-shot learning setup by providing LexGLUE examples in an instruction-following format.

Example from UNFAIR-TOS:

```
Given the following sentence from an online Term of Services:
"if you are a resident of the european union (eu), please note that we offer this alternative 
dispute resolution process, but we can not offer you the european commission dispute platform 
as we do not have an establishment in the eu."

The sentence is unfair with respect to some of the following options:
- Limitation of liability
- Unilateral termination
- Unilateral change
- Content removal
- Contract by using
- Choice of law
- Jurisdiction
- Arbitration
- None

The relevant options are: [None]
```

Example from LEDGAR:

```
Given the following contractual section:
"The corporate law of the State of Delaware shall govern all issues and questions concerning the relative rights of the Corporation and its stockholders. 
All other issues and questions concerning the construction, validity, interpretation and enforcement of this Agreement and the exhibits and schedules hereto 
shall be governed by, and construed in accordance with, the laws of the State of New York, without giving effect to any choice of law or conflict of law rules 
or provisions (whether of the State of New York or any other jurisdiction) that would cause the application of the laws of any 
jurisdiction other than the State of New York."

There is an appropriate section title out of the following options:
- Adjustments
- Agreements
- Amendments
- Anti-Corruption Laws
- Applicable Laws
- Approvals
- Arbitration
- Assignments
- Assigns
...
- Governing Laws
- Waiver Of Jury Trials
- Waivers
- Warranties

The most appropriate option is: [Governing Laws]
```

For all LexGLUE tasks, we provide code to generate a templated instruction-following version of the dataset based on the following template:

```
INTRODUCTORY_PART
"[Input Text]"

OPTIONS_PRESENTATION_TEXT
- [Label 1]
- [Label 2]
...
- [Label L]

QUESTION_TEXT
```
In which case, you can alter the following parts:

```
INTRODUCTORY_PART = 'Given the following sentence from an online Term of Services:'
OPTIONS_PRESENTATION_TEXT = 'The sentence is unfair with respect to some of the following options:'
QUESTION_TEXT = 'The relevant options are:'
```

to generate different templated prompts by updating the `build_instructions/TEMPLATES.py`.

## How to Use and Extend?

You have to follow three easy steps:

1. To generate a templated instruction-following version of a dataset, e.g., EURLEX, you have to call the related script:

```shell
python build_instructions/eurlex_instructions.py
```

2. To call the OpenAI API and get responses (predictions), you have to call the following script:

```shell
python call_openai.py --dataset_name eurlex --model_name gpt-3.5-turbo
Please provide an OpenAI API key: [Copy+Paste API key from https://platform.openai.com/account/api-keys]
```

You may find a list of available models ([here](https://platform.openai.com/docs/models)), so far we provide results for `gpt-3.5-turbo`, which costs $0.002 / 1K tokens. For 1000 examples in LEDGAR, the total number of tokens is 667851, with an estimated processing cost of 1.34$.

3. To evaluate the performance of a GPT-3.5 model, you have to call the following script:

```shell
python evaluate_perfomance.py --dataset_name eurlex --model_name gpt-3.5-turbo --multi_label true
```

## Estimated OpenAI API usage cost for `gpt-3.5-turbo`

| Dataset Name   | Usage Cost | 
|----------------|------------|
 | **ECtHR A**    | $4.43      |
 | **ECtHR B**    | $4.43      | 
 | **SCOTUS**     | $8.49      | 
 | **EURLEX**     | $6.15      | 
 | **LEDGAR**     | $1.34      | 
 | **UNFAIR-ToS** | $0.41      | 
 | **CASEHOLD**   | $0.99      | 
| **Total**      | $26.24     |


## LexGLUE Results

Following [Chalkidis et al. (2022)](https://aclanthology.org/2022.acl-long.297/), we report micro- and macro-F1 scores:

| Dataset Name   | Zero-shot (Baseline Prompts -`gpt-3.5-turbo`) | Supervized (Chalkidis et al., 2022 - `LegalBERT`) |
|----------------|-----------------------------------------------|---------------------------------------------------|
| **ECtHR A**    | 55.3 / 50.6                                   | 70.0 / 64.0                                       |
| **ECtHR B**    | 62.8 / 55.3                                   | 80.4 / 74.7                                       |      
| **SCOTUS**     | 43.8 / 42.0                                   | 76.4 / 66.5                                       |   
| **EURLEX**     | 32.5 / 21.1                                   | 72.1 / 57.4                                       |    
| **LEDGAR**     | 70.1 / 56.7                                   | 88.2 / 83.0                                       |     
| **UNFAIR-ToS** | 41.4 / 22.2                                   | 96.0 / 83.0                                       |      
| **CASEHOLD**   | 59.3 / 59.3                                   | 75.3 / 75.3                                       |       
| **Average**    | 49.0 / 37.1                                   | 78.9 / 70.8                                       |
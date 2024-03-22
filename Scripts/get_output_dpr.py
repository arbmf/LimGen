from xturing.models import BaseModel
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Get output of a model for dpr dataset")

# Define a named argument using a flag
parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-p", "--penalty-alpha", type=float, help="Penalty Alpha")
parser.add_argument("-t", "--max-new-tokens", type=float, help="Max new tokens")


args = parser.parse_args()

print(f"model: {args.model}")
print(f"Penalty Alpha: {args.penalty_alpha}")
print(f"Max new tokens: {args.max_new_tokens}")

model = BaseModel.create(args.model)

data_dict = []

with jsonlines.open("./test.jsonl", mode='r') as reader:
    for row in tqdm(reader):
        data_dict.append(row)

generation_config = model.generation_config()
generation_config.max_new_tokens = args.max_new_tokens
generation_config.do_sample = False
generation_config.penalty_alpha = args.penalty_alpha


prompt_template = """Generate limitations or shortcomings for the following passage from a scientific paper\n passage: \n
{text} 
\n 
Limitations:"""

# prompt for zero shot:
# """Generate one or two main limitations or shortcomings for the a scientific paper\n passage: \n
# {text} 
# \n 
# Limitations:"""

import spacy
nlp = spacy.load("en_core_web_sm")
def split_into_sentences(text):
    """
    Splits the text into sentences using spaCy's sentence boundary detection.
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences
def process_text(text, word_limit=600):
    paragraphs = text.split('\\n')  # Splitting the text into paragraphs
    result = []
    current_set = ""
    current_word_count = 0

    for para in paragraphs:
        words = para.split()
        para_word_count = len(words)

        # If the current paragraph itself exceeds the word limit
        if para_word_count > word_limit:
            if current_set:
                result.append(current_set)
                current_set = ""
                current_word_count = 0

            # Splitting the large paragraph
            for sentence in split_into_sentences(para):
                sentence_len = len(sentence.split())
                if current_word_count + sentence_len > word_limit:
                    result.append(current_set)
                    current_set = ""
                    current_word_count = 0
                current_set += sentence + " "
                current_word_count += sentence_len

        else:
            # Check if adding this paragraph exceeds the word limit
            if current_word_count + para_word_count > word_limit:
                result.append(current_set)
                current_set = para
                current_word_count = para_word_count
            else:
                current_set += '\n' + para if current_set else para
                current_word_count += para_word_count

    if current_set:
        result.append(current_set)

    return result

def get_new_prompt(prompt_template_func, text):
        # replace prompt_template text with the text var
    return prompt_template_func.replace("{text}", text)

import time
from itertools import zip_longest

start = args.start
i = args.start
outputs = []
start_time = time.time()
end_time = time.time()
data_dict = data_dict[start:start+20]
len_data_dict = len(data_dict)
for index in tqdm(range(0,len_data_dict,1)):
    answers = []
    print(f'processing {i} --- prev execution: {end_time - start_time}')
    start_time = time.time()
    paragraphs_arr =[]
    
    if index < len_data_dict:
        paragraphs_arr = process_text(data_dict[index]["content"],350)

    for paragraph in tqdm(paragraphs_arr):
        prompt = get_new_prompt(prompt_template,paragraph)
        output_gen = model.generate(texts=prompt)
        output_gen = output_gen.replace("\n\n","")
        output_gen = output_gen.replace("--","")
        answers.append(output_gen)

     #save the answers to a file


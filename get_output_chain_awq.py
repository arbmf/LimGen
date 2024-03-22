import argparse
from vllm import LLM, SamplingParams


from tqdm import tqdm

# Create an ArgumentParser object and define the arguments
parser = argparse.ArgumentParser(description="Get output of a model")

# Define a named argument using a flag
parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-t", "--max-new-tokens", type=float, help="Max new tokens")
parser.add_argument("-t", "--max-model-length", type=float, help="Max model length")

args = parser.parse_args()


data_dict = []

with jsonlines.open("./test.jsonl", mode='r') as reader:
    for row in tqdm(reader):
        data_dict.append(row)

sampling_params = SamplingParams(temperature=0.8, min_p=0.5, max_tokens=args.max_new_tokens)

llm = LLM(model=args.model, quantization="awq", max_model_len=args.max_model_length, gpu_memory_utilization=0.7, max_context_len_to_capture=args.max_model_length)


#prompt for generating with the full paper trained model
prompt_template =  """Your job is to take in a passage from a research paper and a concise summary of that research and identify one or two main limitation from the given passage using the summary as context.
Paper Passage: \n 
{text} 
\n---------------\n 
Paper Summary: \n 
{summary} 
\n---------------\n 
Limitations:"""

# prompt for generating with DPR trained model can be
# """Your job is to take in a passage from a research paper and a concise summary of that research and write a limitations from the given passage using the summary as context.
# Paper Passage: \n 
# {text} 
# \n---------------\n 
# Paper Summary: \n 
# {summary} 
# \n---------------\n 
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
def process_text(text, word_limit=700):
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


def get_new_prompt(prompt_template_func, text, summary):
        # replace prompt_template text with the text var
    new_prompt = prompt_template_func.replace("{text}", text)
    new_prompt = new_prompt.replace("{summary}",summary)
    return new_prompt

prompts = []


from itertools import zip_longest

len_data_dict = len(data_dict)
for index in tqdm(range(0,len_data_dict,4)):
    answers = [[],[],[],[]]
    start_time = time.time()
    paragraphs_arr =[]
    for k in range(4):
        if index+k < len_data_dict:
            paragraphs_arr.append(process_text(data_dict[index+k]["content"],350))

    prompt_arr = []
    for k in range(4):
        if index+k < len_data_dict:
            prompt_arr.append(get_new_prompt(prompt_template,paragraphs_arr[k].pop(0),data_dict[index+k]["summary"]))

    output_gen = llm.generate(prompt_arr, sampling_params)
    answer_arr = []
    for k in range(4):
        if index+k < len_data_dict:
            answer_arr.append(output_gen[k].outputs[0].text)
            answers[k].append(output_gen[k].outputs[0].text.replace("\n\n",""))

    j=0
    last_answer = [None,None,None,None]
    for paragraphs in zip_longest(*paragraphs_arr, fillvalue=None):
        j +=1
        prompt_arr = []
        for k in range(4):
            if k<len(paragraphs) and paragraphs[k] is not None:
                answer_arr[k] = answer_arr[k].replace("\n","")
                prompt_arr.append(get_new_prompt(prompt_template,paragraphs[k],data_dict[index+k]["summary"]))
                
        output_gen = llm.generate(prompt_arr, sampling_params)
        l = 0
        for k in range(4):
            if k<len(paragraphs) and paragraphs[k] is not None:
                answer_arr[k] = output_gen[l].outputs[0].text
                answer_arr[k] = answer_arr[k].replace("\n\n","")
                answer_arr[k] = answer_arr[k].replace("--","")
                answers[k].append(answer_arr[k])
                last_answer[k] = answer_arr[k]
                l+=1


    #save answers[k]

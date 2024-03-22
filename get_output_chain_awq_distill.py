import argparse



from vllm import LLM, SamplingParams
from tqdm import tqdm


# Create an ArgumentParser object and define the arguments
parser = argparse.ArgumentParser(description="Get output of a model")

# Define a named argument using a flag
parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-t", "--max-new-tokens", type=float, help="Max new tokens")
parser.add_argument("-t", "--max-model-length", type=float, help="Max model length")
parser.add_argument("-t", "--repetition-penalty", type=float, help="Repetition penalty")
args = parser.parse_args()

csv.field_size_limit(sys.maxsize)
data_dict = []

with jsonlines.open("./limgen_candidate_output.jsonl", mode='r') as reader:
    for row in tqdm(reader):
        data_dict.append(row)


sampling_params = SamplingParams(temperature=0.8, min_p=0.5, max_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty)

llm = LLM(model=args.model, quantization="awq", max_model_len=args.max_model_length, gpu_memory_utilization=0.7, max_context_len_to_capture=args.max_model_length)

reduce_template_str = """The following is set of limitations:
{limitations}
Take these and distill it into a final, consolidated list of limitations: 
"""

def get_reduce_template(reduce_template_func, limitations):
    return reduce_template_func.replace("{limitations}", limitations)

prompts = []

len_data_dict = len(data_dict)
for index in tqdm(range(0,len_data_dict,4)):
  
    prompt_arr = []
    for k in range(4):
        if index+k < len_data_dict:
            prompt_arr.append(get_reduce_template(reduce_template_str,data_dict[index+k]["output"]))

    output_gen = llm.generate(prompt_arr, sampling_params)
    answer_arr = [None,None,None,None]
    for k in range(4):
        if index+k < len_data_dict:
            answer_arr[k]=output_gen[k].outputs[0].text.replace("\n\n","")
            answer_arr[k] = answer_arr[k].replace("--","")

    #save answer_arr



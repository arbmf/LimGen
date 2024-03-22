# In[2]:


from xturing.models import BaseModel
import jsonlines
from tqdm import tqdm
import argparse


# Create an ArgumentParser object and define the arguments
parser = argparse.ArgumentParser(description="Get output of a model")

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

#replace with prompt from the paper 
prompt = ""

with jsonlines.open(output_path, mode='r') as reader:
    for row in tqdm(reader):
        #Edit to accomodate the prompt and dataset
        row["model_input"] = replace("{paper_text}", row["content"])
        data_dict.append(row)
print(f"dataset size: {len(data_dict)}")


generation_config = model.generation_config()
generation_config.max_new_tokens = args.max_new_tokens
generation_config.do_sample = False
generation_config.penalty_alpha = args.penalty_alpha


for line in tqdm(data_dict):
    output = model.generate(texts=[line["model_input"]])
    #save the output to a file


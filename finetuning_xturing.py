from pytorch_lightning.loggers import WandbLogger
from xturing.datasets.text_dataset import TextDataset
from xturing.models import BaseModel
from tqdm import tqdm
import jsonlines
import argparse

# Initializes WandB integration 
wandb_logger = WandbLogger(project='project_name')
# Create an ArgumentParser object and define the arguments
parser = argparse.ArgumentParser(description="Get output of a model")

parser.add_argument("-m", "--model", type=str, help="model")
parser.add_argument("-mt", "--max-tokens", type=float, help="Max new tokens")


args = parser.parse_args()

def preprocess_alpaca_json_data_to_text_dataset(dataset_path: str):
    """Creates a dataset in the alpaca format given the JSON dataset.
    :param alpaca_dataset_path: path of the Alpaca dataset
    """
    inputs = []
    outputs = []
    with jsonlines.open(dataset_path, mode='r') as reader:
        for row in tqdm(reader):
            inputs.append("appropriate prompt for fine-tuning(available in the paper)")
            outputs.append(row['limitations'])
    dataset = TextDataset({
        "text": inputs,
        "target": outputs
    })
    return dataset


dataset = preprocess_alpaca_json_data_to_text_dataset("./train.jsonl")
model = BaseModel.create(args.model)




finetuning_config = model.finetuning_config()
finetuning_config.max_length = args.max_tokens
finetuning_config.output_dir = './'
finetuning_config.batch_size = 1
finetuning_config.num_train_epochs = 3
print(model.finetuning_config())

model.finetune(dataset=dataset, logger=wandb_logger)

# Save the model
model.save("./")

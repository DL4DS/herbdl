"""
Script to evaluate the performance of the model on the training data.
"""

import torch

from transformers import (
    AutoTokenizer,
    set_seed,
)
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from PIL import Image
from datasets import load_dataset
import json
import pandas as pd
import os
import logging
import numpy as np

import open_clip

os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='train_evaluation.log')

# Model definition
MODEL_CPKT = "openai/clip-vit-large-patch14-336"
PRETRAINED_MODEL = "dl4ds/herbaria_foundation_model"

model, preprocess, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Dataset
root_csv = "/projectnb/herbdl/data/kaggle-herbaria/train_2022_labeled2.csv"
train_json = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/train_22_scientific.json"
val_json = "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/val_22_scientific.json"

dataset = load_dataset("json", data_files=train_json)

TRAIN_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_metadata.json"

train_metadata = json.load(open(TRAIN_METADATA))
categories = train_metadata['categories']

label_df = pd.read_csv(root_csv)
try:
    label_df.drop(columns=['Unnamed: 0'], inplace=True)
except KeyError:
    pass

categories_df = pd.DataFrame(categories)

def map_label_to_category(label):
    label = label.split(" ")
    species, genus, family = label[2], label[1], label[0]
    category = categories_df[(categories_df['species'] == species) & (categories_df['genus'] == genus) & (categories_df['family'] == family)].iloc[0]['category_id']
    return category

def label_to_taxons(label):
    label = label.split(" ")
    species, genus, family = label[2], label[1], label[0]

    return species, genus, family

def preprocess_image(image_path):
    image = Image.open(image_path)
    
    return preprocess(image).unsqueeze(0)


class ImageDatasetTrain:
    def __init__(self, image_paths, labels):
        self.images = image_paths
        self.labels = labels
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        category = map_label_to_category(label)
        file_name = os.path.basename(path)
        image_id = int(file_name.split(".")[0].split("__")[1])
        return {"pixel_values": preprocess_image(path), "image_id": image_id, "label": label, "category": category, "path": path}

    def sample(self, n):
        indices = np.random.choice(len(self), n, replace=False)
        return [self[i] for i in indices]

train_dataset = ImageDatasetTrain(dataset['train']['image'][:10000], dataset['train']['caption'][:10000])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

logger.info(f"Number of images: {len(train_dataset)}")

taxons = np.array([label_to_taxons(i) for i in dataset['train']['caption']])

species = taxons[:, 0]
genus = taxons[:, 1]
family = taxons[:, 2]

unique_species = sorted(list(set(species)))
unique_genus = sorted(list(set(genus)))
unique_family = sorted(list(set(family)))

logger.info(f"Unique species: {len(unique_species)}")
logger.info(f"Unique genus: {len(unique_genus)}")
logger.info(f"Unique family: {len(unique_family)}")

unique_captions = sorted(list(set(dataset['train']['caption'])))
logger.info(f"Unique captions: {len(unique_captions)}")

text = tokenizer(unique_captions)
text_features = model.encode_text(text)
text_features /= text_features.norm(dim=-1, keepdim=True)


logger.info(f"Label embeddings size: {text_features.size()}")

# model = model.eval()
# model.to(device)

submission_lst = []
images = 0

correct_captions = 0
correct_family = 0
correct_genus = 0
correct_species = 0

species_dict = {species: 0 for species in unique_species}
genus_dict = {genus: 0 for genus in unique_genus}
family_dict = {family: 0 for family in unique_family}

for image in train_dataset:
    path = image['path']
    label = image['label']

    img = preprocess(Image.open(path)).unsqueeze(0)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    prediction = unique_captions[text_probs.argmax().item()]

    if prediction == label:
        correct_captions += 1

        species, genus, family = label_to_taxons(label)

        species_dict[species] += 1
        genus_dict[genus] += 1
        family_dict[family] += 1

        correct_species += 1
        correct_genus += 1
        correct_family += 1

        logger.info("Image: %s, Prediction: %s, Label: %s", path, prediction, label)
    
'''# Write dicts to file
with open("evaluation_result/species_dict.json", "w") as f:
    json.dump(species_dict, f)

with open("evaluation_result/genus_dict.json", "w") as f:
    json.dump(genus_dict, f)

with open("evaluation_result/family_dict.json", "w") as f:
    json.dump(family_dict, f)
'''

logger.info("===== Evaluation Summary =====")
logger.info(f"Correct captions: {correct_captions}")
logger.info(f"Correct family: {correct_family}")
logger.info(f"Correct genus: {correct_genus}")
logger.info(f"Correct species: {correct_species}")
logger.info(f"Total images: {images}")
import torch

from transformers import (
    set_seed,
)
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor
from PIL import Image
from datasets import load_dataset
import json
import pandas as pd
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(42)

# Model definition
MODEL_CPKT = "openai/clip-vit-large-patch14-336"
PRETRAINED_MODEL = "dl4ds/herbaria_foundation_model"

processor = CLIPProcessor.from_pretrained(MODEL_CPKT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()

# Dataset
IMAGE_BASE_DIR = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_images"
TEST_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/test_metadata.json"

metadata = json.load(open(TEST_METADATA))

IMAGE_PATHS = [os.path.join(IMAGE_BASE_DIR, image['file_name']) for image in metadata]

test_annotations = pd.DataFrame(metadata)
test_annotations.head()

root_csv = "/projectnb/herbdl/data/kaggle-herbaria/train_2022_labeled.csv"
out_json = "/projectnb/herbdl/workspaces/smritis/finetuning/training/pairs.json"

dataset = load_dataset("json", data_files=out_json)
index = 0

unique_captions = sorted(list(set(dataset['train']['caption'])))

TRAIN_METADATA = "/projectnb/herbdl/data/kaggle-herbaria/herbarium-2022/train_metadata.json"

train_metadata = json.load(open(TRAIN_METADATA))
categories = train_metadata['categories']

label_df = pd.read_csv(root_csv)
label_df.drop(columns=['Unnamed: 0'], inplace=True)

categories_df = pd.DataFrame(categories)

def map_label_to_category(label):
    label = label.split(" ")
    species, genus, family = label[6][:-1], label[10], label[13][:-1]
    category = categories_df[(categories_df['species'] == species) & (categories_df['genus'] == genus) & (categories_df['family'] == family)].iloc[0]['category_id']
    return category


class ImageDatasetTrain:
    def __init__(self, image_paths, labels, image_processor):
        self.images = image_paths
        self.labels = labels
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        #category = map_label_to_category(label)
        file_name = os.path.basename(path)
        image_id = int(file_name.split(".")[0].split("__")[1])
        return {"pixel_values": self.preprocess_image(path), "image_id": image_id, "label": label, "path": path}

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
    
        return self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze()

    def sample(self, n):
        indices = np.random.choice(len(self), n, replace=False)
        return [self[i] for i in indices]




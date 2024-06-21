import torch

from transformers import (
    AutoTokenizer,
    set_seed,
)
from transformers import CLIPProcessor, CLIPModel
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, PILToTensor
from PIL import Image
from datasets import load_dataset
import torch.multiprocessing as mp
import json
import pandas as pd
import pickle
import os
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "true"
set_seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filemode='w', filename='evaluation.log')

# Model definition
MODEL_CPKT = "openai/clip-vit-large-patch14-336"
PRETRAINED_MODEL = "dl4ds/herbaria_foundation_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_CPKT)
model = CLIPModel.from_pretrained(PRETRAINED_MODEL)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

unique_captions = list(set(dataset['train']['caption']))
print(f"Unique captions: {len(unique_captions)}")

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


# Note: The else block takes a long time
if os.path.exists("preprocessed_images.pkl"):
    with open("preprocessed_images.pkl", "rb") as f:
        preprocessed_images = pickle.load(f)
else:
    transform = Compose([
        Resize((336, 336)),
        PILToTensor(),
    ])

    def preprocess_image(image_path):
        image = Image.open(image_path)
        
        return transform(image).unsqueeze(0)

    preprocessed_images = [{'pixel_values': preprocess_image(image_path), 
    "image_path": image_path} for image_path in IMAGE_PATHS]

print("Number of Preprocessed Images:", len(preprocessed_images))

# Tokenizing captions
tokenized_captions = torch.tensor([tokenizer.encode(caption, padding='max_length') for caption in unique_captions]).cuda()
attention_mask = torch.ones((len(unique_captions), len(tokenized_captions[0]))).cuda()

print("Tokenized Captions Shape:", tokenized_captions.shape)

class CaptionDataset(Dataset):
    def __init__(self, captions, tokenizer):
        self.captions = captions
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]

num_gpus = torch.cuda.device_count()
caption_chunks = [tokenized_captions[i::num_gpus] for i in range(num_gpus)]

model = DataParallel(model).eval().cuda()

submission_lst = []

images_n = 0

def process_chunk(caption_chunk, device, image):
    caption_dataset = CaptionDataset(caption_chunk, processor)
    caption_loader = DataLoader(caption_dataset, batch_size=64, shuffle=False)
    
    max_prob = -float('inf')
    best_caption = None

    for batch in caption_loader:
        
        inputs = {
            "input_ids": batch.to(device),
            "attention_mask": attention_mask,
            "pixel_values": image.to(device),
        }
        
        with torch.no_grad():
            outputs = model.module(**inputs)  # Access the original model in DataParallel
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        batch_max_prob, batch_max_idx = torch.max(probs, dim=1)
        if batch_max_prob > max_prob:
            max_prob = batch_max_prob
            best_caption = caption_chunk[batch_max_idx.item()]

    
    return max_prob, best_caption

images_n = 0
submission_lst = []

if __name__ == '__main__':

    try: 
        
        for image in preprocessed_images:
            pixel_values = image['pixel_values'].cuda()
            image_path = image['image_path']
            image_id = int(image_path.split(".")[0][-1])

            inputs = {"input_ids": tokenized_captions, "pixel_values": pixel_values, "attention_mask": attention_mask}

            with torch.no_grad():
                outputs = model(**inputs)

            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
            
            predicted_label = unique_captions[probs.argmax(dim=1)]
            predicted_category = map_label_to_category(predicted_label)

            submission_lst.append({"Id": image_id, "Predicted": predicted_category})

            images_n += 1
            if images_n % 500 == 0:
                logger.info(f"Processed {images_n} images")

        submission_df = pd.DataFrame(submission_lst)
        submission_df.head()

    except KeyboardInterrupt:
            logger.info(f"Processed {images_n} images")
            pass

submission_df = pd.DataFrame(submission_lst)
submission_df.to_csv("submission.csv", index=False)
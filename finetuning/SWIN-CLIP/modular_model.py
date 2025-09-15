import requests
from PIL import Image
from transformers import VisionTextDualEncoderModel, AutoTokenizer, AutoImageProcessor, Trainer, TrainingArguments, CLIPProcessor
import torch

import pandas as pd
from utils import ImageDatasetTrain

# Initialize the CLIP text encoder and SWIN vision encoder
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
   "microsoft/swin-base-patch4-window7-224-in22k",
   "google-bert/bert-base-uncased"
)

# Initialize the tokenizer and image processor
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/swin-base-patch4-window7-224-in22k")

# Set up datasets and dataloaders
train_df = pd.read_json("../datasets/train_22_scientific_str.json", lines=True)
val_df = pd.read_json("../datasets/val_22_scientific_str.json", lines=True)

top_labels = train_df['caption'].value_counts().nlargest(10).index.tolist()
train_df_sample = train_df[train_df['caption'].isin(top_labels)]
val_df_sample = val_df[val_df['caption'].isin(top_labels)]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384", cache_dir="./tmp/swin")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./tmp/clip")

train_dataset = ImageDatasetTrain(train_df_sample["image"].tolist(), train_df_sample["caption"].tolist(), image_processor)
val_dataset = ImageDatasetTrain(val_df_sample["image"].tolist(), val_df_sample["caption"].tolist(), image_processor)

# Define the training arguments
# Data collator
def data_collator(batch):
    images = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['label'] for item in batch]
    input_ids = tokenizer(text=labels, return_tensors="pt", padding=True).input_ids
    return {"pixel_values": images, "input_ids": input_ids, "labels": labels}

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,
    report_to="none",
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("swin-clip")


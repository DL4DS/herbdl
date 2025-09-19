#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPModel, 
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    CLIPProcessor,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import DataLoader
from utils import ImageDatasetTrain
import pandas as pd
import numpy as np
import argparse
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Parse arguments 
parser = argparse.ArgumentParser(description="Train a SWIN-CLIP model for zero-shot classification")
parser.add_argument("--swin_checkpoint", type=str, default="faridkarimli/SWIN_finetuned_kaggle22", help="SWIN checkpoint to use")
parser.add_argument("--clip_checkpoint", type=str, default="openai/clip-vit-base-patch16", help="CLIP checkpoint to use")
parser.add_argument("--num_labels", type=int, default=50, help="Number of labels to use for training")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model")
parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for training")
parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory to save model checkpoints")
parser.add_argument("--frozen_type", type=str, default="v3", help="Type of frozen layers")


args = parser.parse_args()

FROZEN_TYPE = args.frozen_type


# Load SWIN and CLIP models
clip_model = CLIPModel.from_pretrained(args.clip_checkpoint, cache_dir="./tmp/clip")
vision_backbone = AutoModelForImageClassification.from_pretrained(args.swin_checkpoint, cache_dir="./tmp/swin")

num_gpus = torch.cuda.device_count()

# Define the combined model
class SWIN_CLIP(nn.Module):
    def __init__(self, swin_model, clip_model):
        super(SWIN_CLIP, self).__init__()
        self.swin_model = swin_model
        self.clip_model = clip_model
        self.projection1 = nn.Linear(swin_model.config.hidden_size, clip_model.config.projection_dim)
        self.projection2 = nn.Linear(clip_model.config.projection_dim, clip_model.config.projection_dim)

        for name, param in self.swin_model.named_parameters():
            if FROZEN_TYPE == 'v5':
                    if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3") and not name.startswith("swinv2.encoder.layers.2") and not name.startswith("swinv2.encoder.layers.1") and not name.startswith("swinv2.encoder.layers.0"):
                        param.requires_grad = False
            elif FROZEN_TYPE == "v4":
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3") and not name.startswith("swinv2.encoder.layers.2") and not name.startswith("swinv2.encoder.layers.1"):
                    param.requires_grad = False
            elif FROZEN_TYPE == "v3":
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3") and not name.startswith("swinv2.encoder.layers.2"):
                    param.requires_grad = False
            elif FROZEN_TYPE == "v2":
                if 'classifier' not in name and "swinv2.layernorm" not in name and not name.startswith("swinv2.encoder.layers.3"):
                    param.requires_grad = False
            else:
                if 'classifier' not in name and "swinv2.layernorm" not in name:
                    param.requires_grad = False


    def forward(self, pixel_values, input_ids):
        #print("=== IN FORWARD ===")
        #print(f"Pixel values shape: {pixel_values.shape}")
        #print(f"Input IDs shape: {input_ids.shape}")

        swin_output = self.swin_model(pixel_values, output_hidden_states=True)

        vision_features = self.projection1(swin_output.hidden_states[-1][:, 0, :])  # CLS token

        vision_features = F.relu(self.projection2(F.relu(vision_features)))

        text_outputs = self.clip_model.get_text_features(input_ids)

        # print(f"Vision features shape: {vision_features.shape}")
        # print(f"Text outputs shape: {text_outputs.shape}")

        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(vision_features, text_outputs.T)
        logits = self.clip_model.logit_scale.exp() * similarity
        # print(f"Logits shape: {logits.shape}")

        logits_per_image = logits.T
        #print(f"Logits per image shape: {logits_per_image.shape}")

        # Use self-defined contrastive_loss
        labels = torch.arange(logits.shape[0], device=logits.device)
        # print(f"Labels shape: {labels.shape}")
        loss_img_to_text = F.cross_entropy(logits, labels)
        loss_text_to_img = F.cross_entropy(logits.T, labels)
        loss = (loss_img_to_text + loss_text_to_img) / 2
        
        return loss

class ContrastiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # print("=== IN COMPUTE LOSS ===")
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        loss = model(pixel_values=pixel_values, input_ids=input_ids)
        print(f"Loss shape after model call: {loss.shape}")
        
        '''# Use self-defined contrastive_loss
        labels = torch.arange(logits.shape[0], device=logits.device)
        print(f"Labels shape: {labels.shape}")
        loss_img_to_text = F.cross_entropy(logits, labels)
        loss_text_to_img = F.cross_entropy(logits.T, labels)
        loss = (loss_img_to_text + loss_text_to_img) / 2'''

        avg_loss = torch.mean(loss)        
        return avg_loss

    def compute_loss2(self, model, inputs, return_outputs=False):
        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        logits = model(pixel_values=pixel_values, input_ids=input_ids)
        print(f"Logits shape after model call: {logits.shape}")

        # Determine the batch size per GPU
        batch_size_per_device = logits.size(1)
        
        # Initialize total loss
        total_loss = 0.0
        
        # Compute loss for each GPU's logits
        for i in range(num_gpus):
            # Extract logits for the current GPU
            logits_chunk = logits[:, i * batch_size_per_device:(i + 1) * batch_size_per_device]
            print("Logits chunk shape: ", logits_chunk.shape)
            
            # Generate labels for this chunk (local to the chunk)
            labels_chunk = torch.arange(batch_size_per_device, device=logits.device)
            print(f"Labels chunk shape: {labels_chunk.shape}")
            
            # Compute loss for this chunk
            loss_img_to_text = F.cross_entropy(logits_chunk, labels_chunk)
            loss_text_to_img = F.cross_entropy(logits_chunk.T, labels_chunk)
            total_loss += (loss_img_to_text + loss_text_to_img) / 2


        # Average the loss across all GPUs
        total_loss /= num_gpus
        print(f"Total loss: {total_loss}")
        return total_loss


# Set up datasets and dataloaders
train_df = pd.read_json("../datasets/train_22_scientific_str.json", lines=True)
val_df = pd.read_json("../datasets/val_22_scientific_str.json", lines=True)

top_labels = train_df['caption'].value_counts().nlargest(args.num_labels).index.tolist()
train_df_sample = train_df[train_df['caption'].isin(top_labels)]
val_df_sample = val_df[val_df['caption'].isin(top_labels)]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384", cache_dir="./tmp/swin")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./tmp/clip")

train_dataset = ImageDatasetTrain(train_df_sample["image"].tolist(), train_df_sample["caption"].tolist(), image_processor)
val_dataset = ImageDatasetTrain(val_df_sample["image"].tolist(), val_df_sample["caption"].tolist(), image_processor)

# Define contrastive loss
def contrastive_loss2(logits, labels):
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_img_to_text = F.cross_entropy(logits, labels)
    loss_text_to_img = F.cross_entropy(logits.T, labels)
    return (loss_img_to_text + loss_text_to_img) / 2

# Data collator
def data_collator(batch):
    images = torch.stack([item['pixel_values'] for item in batch])
    labels = [item['label'] for item in batch]
    input_ids = processor(text=labels, return_tensors="pt", padding=True).input_ids
    return {"pixel_values": images, "input_ids": input_ids, "labels": labels}

# Initialize model
model = SWIN_CLIP(vision_backbone, clip_model)

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    evaluation_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=args.weight_decay,
    report_to="none",
    save_strategy="epoch",
    logging_dir="./logs",
)

# Trainer setup
trainer = ContrastiveTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    #compute_metrics=lambda p: {"loss": contrastive_loss(p.predictions, p.label_ids)}
)

# Start training
trainer.train()

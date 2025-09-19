#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import CLIPModel, AutoImageProcessor, AutoModelForImageClassification

from transformers import CLIPProcessor
from torch.utils.data import DataLoader
from utils import ImageDatasetTrain
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
import torch
import torch.nn as nn
from torch.nn import DataParallel

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# parse arguments 
import argparse

parser = argparse.ArgumentParser(description="Train a CLIP model for zero-shot classification")
parser.add_argument("--clip_checkpoint", type=str, default="openai/clip-vit-base-patch16", help="CLIP checkpoint to use")
parser.add_argument("--num_labels", type=int, default=1000, help="Number of labels to use for training")
parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train the model")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for training")
parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory to save model checkpoints")
 
args = parser.parse_args()

NUM_LABELS = args.num_labels
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
OUTPUT_DIR = args.output_dir

######################

CLIP_CPKT = "openai/clip-vit-base-patch16"

print(f"Loading CLIP model {CLIP_CPKT}")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./tmp/clip")

model = CLIPModel.from_pretrained(CLIP_CPKT, cache_dir="./tmp/clip") #, attn_implementation="flash_attention_2")
learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {learnable_params}")

train_df = pd.read_json("../datasets/train_22_scientific_str.json", lines=True)
val_df = pd.read_json("../datasets/val_22_scientific_str.json", lines=True)

print(f"Training dataset size: {train_df.shape}, Validation dataset size: {val_df.shape}. Columns: {train_df.columns}" )

np.random.seed(42)

top_labels = train_df['caption'].value_counts().nlargest(NUM_LABELS).index.tolist()
train_df_sample = train_df[train_df['caption'].isin(top_labels)]
val_df_sample = val_df[val_df['caption'].isin(top_labels)]


image_paths = train_df_sample["image"].tolist()
labels = train_df_sample["caption"].tolist()

val_image_paths = val_df_sample["image"].tolist()
val_labels = val_df_sample["caption"].tolist()

print(f"Training set size: {train_df_sample.shape}")
print(f"Validation set size: {val_df_sample.shape}")

train_dataset = ImageDatasetTrain(image_paths, labels, processor)
val_dataset = ImageDatasetTrain(val_image_paths, val_labels, processor)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

def contrastive_loss(logits, labels):
    # Symmetric cross-entropy loss for image-to-text and text-to-image
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_img_to_text = F.cross_entropy(logits, labels)
    loss_text_to_img = F.cross_entropy(logits.T, labels)
    return (loss_img_to_text + loss_text_to_img) / 2

val_labels = set()
for batch in val_dataloader:
    labels = batch['label']
    val_labels.update(labels)

val_labels = list(val_labels)
val_labels_processed = processor(text=val_labels, return_tensors="pt", padding=True).input_ids.to("cuda")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def validate(model, val_dataloader, device, verbose=False):
    model.eval()
    correct = 0
    top5 = 0
    top10 = 0

    for batch in val_dataloader:
        correct_label = batch['label'][0]
        pixel_values = batch['pixel_values'].to(device)

        with torch.no_grad():
            output = model(val_labels_processed, pixel_values)
            logits_img = output.logits_per_image

        predicted_label_idx = logits_img.argmax().item()

        predicted_label = val_labels[predicted_label_idx]
        correct_label_idx = val_labels.index(correct_label)

        if predicted_label == correct_label:
            if verbose:
                print(f"Correct: {correct_label}, Predicted: {predicted_label}. Number of correct predictions: {correct}/{len(val_dataset)}")
            correct += 1
        
        elif correct_label_idx in logits_img.topk(5).indices:
            top5 += 1

        elif correct_label_idx in logits_img.topk(10).indices:
            top10 += 1
        

    return correct, top5, top10



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_devices = torch.cuda.device_count()
print(f"Using {n_devices} GPUs")

# model = DataParallel(model)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=EPOCHS)

losses = []
eval_accs = []

print('##########################')
print("Starting training...")

for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        labels = batch['label']
        pixel_values = batch['pixel_values'].to(device)
        input_ids = processor(text=labels, return_tensors="pt", padding=True).input_ids.to(device)

        output = model(input_ids, pixel_values)
        logits_img = output.logits_per_image
        logits_text = output.logits_per_image
        # print(f"Logits shape: {logits_img.shape}, {logits_text.shape}")
        # print(f"Labels shape: {len(labels)}")
        
        # print((logits_img == logits_text).all())

        loss = contrastive_loss(logits_img, labels)
        loss.backward()    

        total_loss += loss.item()

        optimizer.step()

    avg_loss = total_loss / BATCH_SIZE
    losses += [avg_loss]
    total_loss = 0
    lr_scheduler.step()

    if epoch % 10 == 0 and epoch > 0:
        correct, top5, top10 = validate(model, val_dataloader, device)

        print("--------------------")
        print(f"Epoch: {epoch}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Top 1 accuracy: {correct / len(val_dataset) * 100:.2f}%")
        print("--------------------")

        eval_accs += [correct / len(val_dataset) * 100]

    # if the last 5 epochs have not shown improvement, increase the learning rate
    if len(eval_accs) > 5 and np.allclose(eval_accs[-4:], [eval_accs[-1]], atol=0.1):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 2
        print(f"Learning rate increased to {param_group['lr']}")

            
# save model

model_name = f"baseline_{len(val_labels)}_labels_{EPOCHS}_epochs"
path = f"./checkpoints/{model_name}"
os.makedirs(f"./checkpoints/{model_name}", exist_ok=True)

results = {
    "losses": losses,
    "eval_accs": eval_accs,
    "final_eval_acc": eval_accs[-1],
    "num_labels": NUM_LABELS,
    "num_learnable_params": learnable_params
}
with open(f"./checkpoints/{model_name}/results.json", "w") as f:
    json.dump(results, f)

torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-1],
            }, f"./checkpoints/{model_name}/model.pth")

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(eval_accs, marker="o")
plt.savefig(f"./checkpoints/{model_name}/plot.png")

train_labels = set()

for batch in train_dataloader:
    labels = batch['label']
    train_labels.update(labels)

train_labels = list(train_labels)
print(f"Length of train labels: {len(train_labels)}")



val_labels = set()
device="cuda"
for batch in val_dataloader:
    labels = batch['label']
    val_labels.update(labels)

all_labels = list(val_labels)
    
all_labels_processed = processor(text=all_labels, return_tensors="pt", padding=True).input_ids.to(device)
all_labels_processed.shape


## Zero-shot classification
np.random.seed(42)
NUM_ZEROSHOT_LABELS = 50
zeroshot_labels_sample = np.random.choice(train_df[~train_df['caption'].isin(top_labels)]['caption'].unique(), NUM_ZEROSHOT_LABELS, replace=False)
zeroshot_sample = train_df[train_df['caption'].isin(zeroshot_labels_sample)]

all_zeroshot_labels = list(zeroshot_sample['caption'].unique())
all_zeroshot_labels_processed = processor(text=all_zeroshot_labels, return_tensors="pt", padding=True).input_ids.to(device)

zerosho_image_paths = zeroshot_sample["image"].tolist()
zeroshot_labels = zeroshot_sample["caption"].tolist()

zeroshot_dataset = ImageDatasetTrain(zerosho_image_paths, zeroshot_labels, processor)
zeroshot_dataloader = DataLoader(zeroshot_dataset, batch_size=1, shuffle=False)

print(f"Number of zeroshot labels: {NUM_ZEROSHOT_LABELS}")
print(f"Length of zeroshot dataset: {len(zeroshot_dataset)}")

model.to(device)
model.eval()

correct = 0
top5 = 0

for batch in tqdm(zeroshot_dataloader, desc="Zero-shot validation"):
    correct_label = batch['label'][0]
    pixel_values = batch['pixel_values'].to(device)

    with torch.no_grad():
        logits = model(all_zeroshot_labels_processed, pixel_values).logits_per_image

    predicted_label_idx = logits.argmax().item()

    predicted_label = all_zeroshot_labels[predicted_label_idx]
    correct_label_idx = all_zeroshot_labels.index(correct_label)
    #print(f"Predicted label: {predicted_label}. Correct label: {correct_label}")

    if predicted_label == correct_label:
        correct += 1
        top5 += 1
    elif correct_label_idx in logits.topk(5).indices:
        # print(f"Predicted label: {predicted_label}. Correct label: {correct_label}")
        # print(f"Top 5 labels: {[all_zeroshot_labels[x] for x in logits.topk(5).indices[0]]}")
        top5 += 1

print(f"Zero-shot top 1 accuracy: {correct / len(zeroshot_dataset) * 100:.2f}%")
print(f"Zero-shot top 5 accuracy: {top5 / len(zeroshot_dataset) * 100:.2f}%")

# Write the results to a file

results = {
    "zero_shot_top1": correct / len(zeroshot_dataset) * 100,
    "zero_shot_top5": top5 / len(zeroshot_dataset) * 100,
    "num_zero_shot_labels": NUM_ZEROSHOT_LABELS
}

with open(f"./checkpoints/{model_name}/zero_shot_results.json", "w") as f:
    json.dump(results, f)




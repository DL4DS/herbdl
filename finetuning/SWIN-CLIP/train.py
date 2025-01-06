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

parser = argparse.ArgumentParser(description="Train a SWIN-CLIP model for zero-shot classification")
parser.add_argument("--swin_checkpoint", type=str, default="faridkarimli/SWIN_finetuned_kaggle22", help="SWIN checkpoint to use")
parser.add_argument("--clip_checkpoint", type=str, default="openai/clip-vit-base-patch16", help="CLIP checkpoint to use")
parser.add_argument("--num_labels", type=int, default=22, help="Number of labels to use for training")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the model")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for training")
parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory to save model checkpoints")
parser.add_argument("--model_type", type=str, default="base", help="Type of model (base or finetuned)")
parser.add_argument("--freeze_type", type=str, default="v2", help="Freeze type (v0: freeze all layers, v1: freeze all but the last layer, v2: freeze all but the last classifier head and last transformer layer, v3: freeze all but the last classifier head and last 2 transformer layers)")

args = parser.parse_args()

NUM_LABELS = args.num_labels
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
WEIGHT_DECAY = args.weight_decay
OUTPUT_DIR = args.output_dir
MODEL_TYPE = args.model_type
FROZEN_TYPE = args.freeze_type

######################

SWIN_FINETUNED = "faridkarimli/SWIN_finetuned_kaggle22"
SWIN_22K_B = "microsoft/swinv2-base-patch4-window12-192-22k"
SWIN_22K_L = "microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft"

if MODEL_TYPE == "base":
    MODEL_CPKT=SWIN_22K_B
else:
    MODEL_CPKT=SWIN_FINETUNED

CLIP_CPKT = "openai/clip-vit-base-patch16"

clip_model = CLIPModel.from_pretrained(CLIP_CPKT, cache_dir="./tmp/clip")
vision_backbone = AutoModelForImageClassification.from_pretrained(MODEL_CPKT, cache_dir="./tmp/swin")

print(f"Loading SWIN model {MODEL_CPKT}")
print(f"Loading CLIP model {CLIP_CPKT}")

# swin_model.load_state_dict(torch.load("/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/checkpoint-70-epochs/pytorch_model.bin"))

# Add a projection layer if needed to match CLIP's embedding size
class SWIN_CLIP(nn.Module):
    def __init__(self, swin_model, clip_model):
        super(SWIN_CLIP, self).__init__()
        self.swin_model = swin_model
        self.clip_model = clip_model

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
        
        # Add a linear layer to project SWIN embeddings to CLIP's dimension (e.g., 512)
        self.projection1 = nn.Linear(swin_model.config.hidden_size, clip_model.config.projection_dim)
        self.projection2 = nn.Linear(clip_model.config.projection_dim, clip_model.config.projection_dim)
        

    def forward(self, pixel_values, input_ids):
        # Forward pass through SWIN backbone
        try:
            swin_output = self.swin_model(pixel_values)
            vision_outputs = swin_output.last_hidden_state
        except AttributeError:
            swin_output = self.swin_model(pixel_values, output_hidden_states=True)
            vision_outputs = swin_output.hidden_states[-1]
        
        vision_features = self.projection1(vision_outputs[:, 0, :])  # CLS token
        vision_features = F.relu(vision_features)

        vision_features = self.projection2(vision_features)
        vision_features = F.relu(vision_features)
        
        # Forward pass through CLIP's text encoder
        text_outputs = self.clip_model.get_text_features(input_ids)

        # Normalize the vision and text features to compute cosine similarity
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        # print(f"Vision features shape: {vision_features.shape}")
        # print(f"Text outputs shape: {text_outputs.shape}")

        # Compute similarity (cosine similarity is just the dot product after normalization)
        similarity = torch.matmul(vision_features, text_outputs.T)

        # logit_scale is typically initialized as a learnable parameter
        logit_scale = self.clip_model.logit_scale.exp()  # logit_scale starts around 2.6592 in CLIP

        # Multiply similarity scores by logit scale
        logits = logit_scale * similarity

        # print(f"Logits shape: {logits.shape}")

        return logits




processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", cache_dir="./tmp/clip")
image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384", cache_dir="./tmp/swin")

model = SWIN_CLIP(vision_backbone, clip_model)
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
print(f"Training set size: {train_df_sample.shape}")
print(f"Validation set size: {val_df_sample.shape}")

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-patch4-window12-384", cache_dir="./tmp/swin")

train_dataset = ImageDatasetTrain(image_paths, labels, image_processor)
val_dataset = ImageDatasetTrain(val_df_sample["image"].tolist(), val_df_sample["caption"].tolist(), image_processor)

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
            logits = model(pixel_values, val_labels_processed)

        predicted_label_idx = logits.argmax().item()

        predicted_label = val_labels[predicted_label_idx]
        correct_label_idx = val_labels.index(correct_label)

        if predicted_label == correct_label:
            if verbose:
                print(f"Correct: {correct_label}, Predicted: {predicted_label}. Number of correct predictions: {correct}/{len(val_dataset)}")
            correct += 1
        
        elif correct_label_idx in logits.topk(5).indices:
            top5 += 1

        elif correct_label_idx in logits.topk(10).indices:
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

for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        labels = batch['label']
        pixel_values = batch['pixel_values'].to(device)
        input_ids = processor(text=labels, return_tensors="pt", padding=True).input_ids.to(device)

        output = model(pixel_values, input_ids)

        loss = contrastive_loss(output, labels)
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

model_name = f"{MODEL_TYPE}_{len(val_labels)}_labels_{EPOCHS}_epochs_{FROZEN_TYPE}"
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
plt.show()


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
zeroshot_labels_sample = np.random.choice(train_df[~train_df['caption'].isin(top_labels)]['caption'].unique(), 50, replace=False)
zeroshot_sample = train_df[train_df['caption'].isin(zeroshot_labels_sample)]

all_zeroshot_labels = list(zeroshot_sample['caption'].unique())
all_zeroshot_labels_processed = processor(text=all_zeroshot_labels, return_tensors="pt", padding=True).input_ids.to(device)

zeroshot_dataset = ImageDatasetTrain(zeroshot_sample["image"].tolist(), zeroshot_sample["caption"].tolist(), image_processor)
print(f"Number of zeroshot labels: {NUM_ZEROSHOT_LABELS}")
print(f"Length of zeroshot dataset: {len(zeroshot_dataset)}")
zeroshot_dataloader = DataLoader(zeroshot_dataset, batch_size=1, shuffle=False)


model.to(device)
model.eval()

correct = 0
top5 = 0

for batch in tqdm(zeroshot_dataloader, desc="Zero-shot validation"):
    correct_label = batch['label'][0]
    pixel_values = batch['pixel_values'].to(device)

    with torch.no_grad():
        logits = model(pixel_values, all_zeroshot_labels_processed)

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




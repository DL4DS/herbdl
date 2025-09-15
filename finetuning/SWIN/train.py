"""
This script is used to fine-tune a partially frozen SWIN model on a custom dataset.
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel as DP
from torch.utils.data import DataLoader
from tqdm import tqdm


from transformers import AutoModelForImageClassification, AutoImageProcessor

from utils import ImageDatasetTrain

MODEL_PATH="../output/SWIN/kaggle22/" # local path to the model
MODEL_NAME="SWIN_finetuned_kaggle22"
MODEL_CPKT = "microsoft/swin-base-patch4-window12-384"

torch.cuda.empty_cache()

model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
image_processor = AutoImageProcessor.from_pretrained(MODEL_CPKT)

print(f"Parameter names: {[name for name, _ in model.named_parameters()]}")

parameters = model.named_parameters()


print("Freezing every layer except for layers.3, classifier and final layernorm")
trainable = 0
for param in parameters:
    name = param[0]
    if 'classifier' not in name and "swinv2.layernorm" not in name:
        param[1].requires_grad = False
    else:
        trainable += 1

print(f"Trainable parameters: {trainable}")

#########################
# Load dataset
#########################

train_df = pd.read_json("../datasets/train_22_scientific.json", lines=True)
val_df = pd.read_json("../datasets/val_22_scientific.json", lines=True)

image_paths = train_df["image"].tolist()
labels = train_df["caption"].tolist()

val_paths = val_df["image"].tolist()
val_labels = val_df["caption"].tolist()

train_dataset = ImageDatasetTrain(image_paths, labels, image_processor)
val_dataset = ImageDatasetTrain(val_paths, val_labels, image_processor)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Train size: {train_df.size}, Val size: {val_df.size}. Columns: {train_df.columns}" )

#################
# Training
#################

num_cuda_devices = torch.cuda.device_count()
print(f'Number of CUDA devices: {num_cuda_devices}')

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

checkpoint_dir = "./frozen_checkpoints"

def train_model(model, train_dataloader, val_dataloader, epochs, device):
    model = DP(model)
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
            images = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()  
            
            outputs = model(images).logits

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
        
        avg_train_loss = running_loss / len(train_dataloader)
        train_accuracy = 100. * correct_train / total_train
        
        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                
                # Accumulate validation loss and accuracy
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_val += predicted.eq(labels).sum().item()
                total_val += labels.size(0)
        
        avg_val_loss = running_val_loss / len(val_dataloader)
        val_accuracy = 100. * correct_val / total_val
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Optional: Save checkpoint if validation accuracy improves (or periodically)
        # Save the model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')

        # Manage the number of checkpoints: keep only the last 5
        checkpoints = sorted(os.listdir(checkpoint_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        while len(checkpoints) > 5:
            oldest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
            os.remove(oldest_checkpoint)  # Remove the oldest checkpoint
            checkpoints.pop(0)  # Remove the filename from the list

# Usage
train_model(model, train_dataloader, val_dataloader, epochs=10, device='cuda')



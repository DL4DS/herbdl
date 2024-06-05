#!/bin/bash -l

#$ -l gpus=1

# Directory to search
dir="/projectnb/herbdl/workspaces/smritis/finetuning/output/finetuned-kaggle-2022"

# Find the most recent file
rc=$(ls -t "$dir"/* | head -n 1)

recent_file=${rc::-1}

python CLIP_finetuning.py \
    --output_dir $dir \
    --resume_from_checkpoint $recent_file \
    --model_name_or_path "openai/clip-vit-large-patch14-336" \
    --train_file "/projectnb/herbdl/workspaces/smritis/finetuning/training/pairs.json" \
    --image_column image \
    --overwrite_output_dir=True \
    --max_seq_length=77 \
    --num_train_epochs=1 \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --per_device_train_batch_size=8 \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1


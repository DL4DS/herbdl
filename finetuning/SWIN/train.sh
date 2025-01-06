#!/bin/bash -l
#$ -N SWIN_fr_v5_2

module load miniconda
module load academic-ml/spring-2024

#set env variable
export LR_TYPE="linear"
export FROZEN="true"
export FROZEN_TYPE="v5" # v1 - freeze all except last linear layer, # v2 - all layers except last transformer block and linear layer, v3 - all layers except last 2 transformer blocks and linear layer, v4 - all layers except last 3 transformer blocks and linear layer
export RUN_GROUP="SWIN_frozen_v5_2"
export RUN_NAME="SWIN_frozen_v5_2"
export RUN_ID="swin_frozen_v5_1114"

conda activate farid-2024

save_dir="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/frozenv5"

python SWIN_finetuning.py \
    --output_dir  $save_dir \
    --logging_dir $save_dir \
    --model_name_or_path "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/frozenv3/checkpoint-21008"\
    --train_file "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/train_22_scientific.json" \
    --validation_file "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/val_22_scientific.json" \
    --image_column_name image \
    --label_column_name caption \
    --max_seq_length=15 \
    --num_train_epochs=15 \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --learning_rate=0.001 --save_strategy="epoch" --save_total_limit=5 --evaluation_strategy="epoch" \
    --lr_scheduler_type=$LR_TYPE \
    --ignore_mismatched_sizes --overwrite_output_dir --report_to="wandb" \
    --push_to_hub --hub_token="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" --hub_model_id="faridkarimli/SWIN_finetuned_frozen_v5_2"

# 
# --resume_from_checkpoint="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-139125" \
# qsub -l h_rt=36:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -m beas -M faridkar@bu.edu train.sh

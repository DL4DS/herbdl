#!/bin/bash -l

#set env variable
export LR_TYPE="linear"
export FROZEN="true"
export FROZEN_TYPE="v5" # v1 - freeze all except last linear layer, # v2 - all layers except last transformer block and linear layer, v3 - all layers except last 2 transformer blocks and linear layer, v4 - all layers except last 3 transformer blocks and linear layer
export RUN_GROUP="gaudi_test"
export RUN_NAME="SWIN_Gaudi_test"
export RUN_ID="swin_gaudi_test"

save_dir="./output/gaudi_test"

PT_HPU_LAZY_MODE=0 python SWIN_finetuning.py \
    --output_dir  $save_dir \
    --logging_dir $save_dir \
    --model_name_or_path "microsoft/swinv2-large-patch4-window12-192-22k" \
    --train_file "/mnt/purenfs/projects/herbdl/finetuning/train_22_encoded.json" \
    --validation_file "/mnt/purenfs/projects/herbdl/finetuning/val_22_encoded.json" \
    --image_column_name image \
    --label_column_name caption \
    --max_seq_length=15 \
    --num_train_epochs=5 \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --learning_rate=0.001 --save_strategy="epoch" --save_total_limit=5 --evaluation_strategy="epoch" \
    --lr_scheduler_type=$LR_TYPE \
    --ignore_mismatched_sizes --overwrite_output_dir --report_to="wandb" \
    --push_to_hub --hub_token="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" --hub_model_id="faridkarimli/SWIN_Gaudi_test" \
    --gaudi_config_name Habana/swin \
    --use_habana \
    --use_lazy_mode
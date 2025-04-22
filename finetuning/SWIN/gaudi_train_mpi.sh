#!/bin/bash -l

#set env variable
export LR_TYPE="linear"
export FROZEN="false"
export FROZEN_TYPE="v5" # v1 - freeze all except last linear layer, # v2 - all layers except last transformer block and linear layer, v3 - all layers except last 2 transformer blocks and linear layer, v4 - all layers except last 3 transformer blocks and linear layer
export RUN_GROUP="gaudi_mpi"
export RUN_NAME="SWIN_L_384"
export RUN_ID="swin_gaudi_mpi_384"

# log the env variables
echo "--------------------------------"
echo "Running SWIN_L_384 with MPI. Environment variables:"
echo "LR_TYPE: $LR_TYPE"
echo "FROZEN: $FROZEN"
echo "FROZEN_TYPE: $FROZEN_TYPE"
echo "RUN_GROUP: $RUN_GROUP"
echo "RUN_NAME: $RUN_NAME"
echo "RUN_ID: $RUN_ID"
echo "--------------------------------"

save_dir="./output/swin_L_384"

echo "Saving to: $save_dir"

PYTHONUNBUFFERED=1 PT_HPU_LAZY_MODE=0 python gaudi_spawn.py \
    --world_size 8 --use_mpi SWIN_finetuning.py \
    --output_dir  $save_dir \
    --logging_dir $save_dir \
    --model_name_or_path "microsoft/swin-large-patch4-window12-384-in22k" \
    --train_file "/mnt/purenfs/projects/herbdl/finetuning/train_22_encoded.json" \
    --validation_file "/mnt/purenfs/projects/herbdl/finetuning/val_22_encoded.json" \
    --image_column_name image \
    --label_column_name caption \
    --max_seq_length=15 \
    --num_train_epochs=100 \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=128 \
    --learning_rate=0.0005 --save_strategy="epoch" --save_total_limit=5 --evaluation_strategy="epoch" \
    --lr_scheduler_type=$LR_TYPE \
    --ignore_mismatched_sizes --overwrite_output_dir --report_to="wandb" \
    --push_to_hub --hub_token="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" --hub_model_id="faridkarimli/SWIN_Gaudi_100" \
    --gaudi_config_name Habana/swin \
    --use_habana \
    --use_lazy_mode=0 \
    --torch_compile_backend hpu_backend \
    --torch_compile \


# --resume_from_checkpoint "/mnt/purenfs/projects/herbdl/finetuning/SWIN/output/gaudi_mpi/checkpoint-19710" \
#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

save_dir="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22"

python SWIN_finetuning.py \
    --output_dir  $save_dir \
    --logging_dir $save_dir \
    --model_name_or_path "microsoft/swinv2-base-patch4-window12-192-22k" \
    --train_file "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/train_22_scientific.json" \
    --validation_file "/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/datasets/val_22_scientific.json" \
    --image_column_name image \
    --label_column_name caption \
    --max_seq_length=15 \
    --num_train_epochs=45 \
    --remove_unused_columns=False \
    --do_train \
    --do_eval \
    --per_device_train_batch_size=128 \
    --per_device_eval_batch_size=4 \
    --learning_rate=0.005 --save_strategy="epoch" --save_total_limit=5 --evaluation_strategy="epoch" --save_total_limit=5 \
    --lr_scheduler_type="cosine" \
    --ignore_mismatched_sizes --report_to="wandb" --overwrite_output_dir --resume_from_checkpoint="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-45955"

# qsub -l h_rt=36:00:00 -pe omp 32 -P herbdl -l gpus=4 -l gpu_c=8.0 -m beas -M faridkar@bu.edu train.sh


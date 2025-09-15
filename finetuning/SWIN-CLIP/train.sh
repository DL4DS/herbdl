#!/bin/bash -l
#$ -N SWCL-1k-f-v3

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python train.py --num_labels=1000 --epochs=80 --batch_size=128 --lr=1e-4 --model_type="finetuned" --freeze_type="v2"

# --resume_from_checkpoint="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-139125" \
# qsub -l h_rt=48:00:00 -pe omp 16 -P herbdl -l gpus=2 -l gpu_c=8.0 -m beas -M faridkar@bu.edu train.sh

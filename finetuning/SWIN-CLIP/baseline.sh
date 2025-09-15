#!/bin/bash -l
#$ -N CLIP-1k

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python train_baseline.py

# --resume_from_checkpoint="/projectnb/herbdl/workspaces/faridkar/herbdl/finetuning/output/SWIN/kaggle22/checkpoint-139125" \
# qsub -l h_rt=36:00:00 -pe omp 16 -P herbdl -l gpus=4 -l gpu_c=8.0 -m beas -M faridkar@bu.edu baseline.sh

#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python train_evaluation.py

# qsub -l h_rt=36:00:00 -pe omp 32 -P herbdl -l gpus=4 -l gpu_type=A100 -l gpu_memory=80 -m beas -M faridkar@bu.edu job.sh
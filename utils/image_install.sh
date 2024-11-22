#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate farid-2024

python image_install_parallel.py

### The command below is used to submit the job to the cluster
### qsub -N image_install -l h_rt=36:00:00 -pe omp 16 -P herbdl -m beas -M faridkar@bu.edu image_install.sh

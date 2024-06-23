#!/bin/bash -l

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

export PUSHOVER_API_TOKEN=a56w88zdckxutyn1kvgjae4mtn94cj
export PUSHOVER_USER_KEY=um1aa9q3cyj4qcyb3johyu3i3es8d9

python image_install_parallel.py

### The command below is used to submit the job to the cluster
### qsub -l h_rt=24:00:00 -pe omp 8 -P herbdl -m beas -M faridkar@bu.edu image_install.sh

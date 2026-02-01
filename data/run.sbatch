#!/bin/bash
#SBATCH --job-name=data_split
#SBATCH --output=slurm-splitting-%j.out
#SBATCH --error=slurm-splitting-%j.err
#SBATCH -p cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=1-00:00:00
#SBATCH -c 40
#SBATCH --mem=100G

activate base 
# srun python ./sapiens/post_processing/split_data.py './configs/data_config.json'
srun python scfoundation_data_prepare.py 
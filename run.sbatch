#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --time=48:00:00
#SBATCH --mem=100GB
#SBATCH --job-name='yl7276_1012'
#SBATCH --mail-user=yl7276@nyu.edu
#SBATCH --output=slurm_%j.out

module purge
module load anaconda3/5.3.1
module load cuda/10.0.130
module load gcc/6.3.0

source activate /scratch/yl7276/nlu/env
source activate tf-2.0

cd
cd /scratch/yl7276/1012_proj/
python preprocess.py> preprocess.out
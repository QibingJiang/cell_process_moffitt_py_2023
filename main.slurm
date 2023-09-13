#!/bin/bash
#SBATCH --cpus-per-task=70
#SBATCH --time=240:00:00
#SBATCH --error=myjobresults-%J.err
#SBATCH --output=myjobresults-%J.out
#SBATCH --job-name=qibing_task
#SBATCH --mem=300000

python3 ./main.py
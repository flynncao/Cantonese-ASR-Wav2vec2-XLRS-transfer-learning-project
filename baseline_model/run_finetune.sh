#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G  # 增加内存请求

#SBATCH --job-name=train_fs_slurm_pre_cantonese_job
#SBATCH --output=train_fs_slurms_pre_cantonese_job.out
#SBATCH --error=train_fs_slurm_pre_cantonese_job.err

source /scratch/s5600502/thesis_project/asr_cantonese/bin/activate


cd /scratch/s5600502/thesis_project/baseline
python fine_tune_xlsr_wav2vec2_on_cantonese.py

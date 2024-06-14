#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=v100
#SBATCH --mem=64G  # 增加内存请求

#SBATCH --job-name=train_fs_slurm_pre_cantonese_job
#SBATCH --output=train_fs_slurms_pre_cantonese_job.out
#SBATCH --error=train_fs_slurm_pre_cantonese_job.err

source /scratch/s5600502/thesis_project/asr_cantonese/bin/activate


cd /scratch/s5600502/thesis_project/mandarin_cantonese
python fine-tune_mandarin_wav2vec2_cantonese.py --model_dir /scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-model --processor_dir /scratch/s5600502/thesis_project/mandarin_cantonese/cantonese-tokenized-wav2vec2-processor

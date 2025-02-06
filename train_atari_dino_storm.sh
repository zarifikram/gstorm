#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:a100:1
#SBATCH -t 2-12:00:00
#SBATCH --cpus-per-task 18
#SBATCH --mem=80G                                        # Ask for 10 GB of RAM

module load python/3.10

source $HOME/scratch/storm/bin/activate

mkdir logs

#Loading modules

env_name=$1
seed=$2
p_noise=$3
exp_name=${env_name}-DINOSTORM-100k-seed_${seed}_p_noise_${p_noise}
wandb_exp_name=gitstorm_${env_name}_seed${seed}_pnoise${p_noise}
proj_name=BS-STORM-Test 
python train_atari.py  --config-name DINOSTORM_Atari \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/DINOSTORM_Atari.yaml" \
BasicSettings.env_name="ALE/${env_name}-v5" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
Ablation.p_noise=${p_noise} \
wandb.exp_name=${wandb_exp_name} \
wandb.project_name=${proj_name} 
# > 'logs/'${exp_name}'.out' \
# 2> 'logs/'${exp_name}'.err'

# srun --partition=long --gpus=1 --cpus-per-task=1 --time=02:00:00 --pty bash -i

#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task 18
#SBATCH --gpus 1 
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM

module load python/3.10

source $HOME/scratch/storm/bin/activate

mkdir logs

#Loading modules

env_name=$1
seed=$2
exp_name=${env_name}-STORM-life_done-wm_2L512D8H-100k-seed_${seed}
wandb_exp_name=storm_${env_name}_seed${seed}
proj_name=BS-STORM-Test
nohup python train_atari.py \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/STORM_Atari.yaml" \
BasicSettings.env_name="ALE/${env_name}-v5" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
wandb.exp_name=${wandb_exp_name} \
wandb.project_name=${proj_name} \
> 'logs/'${exp_name}'.out' \
2> 'logs/'${exp_name}'.err'

# srun --partition=long --gpus=1 --cpus-per-task=1 --time=02:00:00 --pty bash -i

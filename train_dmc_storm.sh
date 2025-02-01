#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 22:00:00
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1 
#SBATCH --mem=64G                                        # Ask for 10 GB of RAM

module load python/3.10

source $HOME/scratch/storm/bin/activate

mkdir logs

#Loading modules
num_envs=16
env_name=$1
task_name=$2
seed=$3

exp_name=${env_name}_${task_name}-STORM-life_done-wm_2L512D8H-1M-seed_${seed}
wandb_exp_name=storm_${env_name}${task_name}_seed${seed}
proj_name=BS-STORM-Test
MUJOCO_GL=egl nohup python train_dmc.py  --config-name STORM_DMC  \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/STORM_DMC.yaml" \
BasicSettings.env_name="${env_name}" \
BasicSettings.task_name="${task_name}" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
JointTrainAgent.NumEnvs=${num_envs} \
JointTrainAgent.TrainDynamicsEverySteps=${num_envs} \
JointTrainAgent.TrainAgentEverySteps=${num_envs} \
wandb.exp_name=${wandb_exp_name} \
wandb.project_name=${proj_name} \
> 'logs/'${exp_name}'.out' \
2> 'logs/'${exp_name}'.err'
# MUJOCO_GL=egl nohup python train_dmc.py \
# BasicSettings.n=${exp_name} \
# BasicSettings.Seed=${seed} \
# BasicSettings.config_path="config_files/DMC.yaml" \
# BasicSettings.env_name="${env_name}" \
# BasicSettings.task_name="${task_name}" \
# BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
# JointTrainAgent.SampleMaxSteps=${steps} \
# Models.MaskGit.M=${m} \
# Models.MaskGit.T_revise=${T_revise} \
# Models.MaskGit.T_draft=${T_draft} \
# > 'logs/'${exp_name}'.out' \
# 2> 'logs/'${exp_name}'.err'

# srun --partition=gpu_h100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=02:00:00 --pty bash -i

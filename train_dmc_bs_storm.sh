#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-22:00:00
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1 
#SBATCH --mem=16G                                        # Ask for 10 GB of RAM

module load python/3.10

source $HOME/scratch/storm/bin/activate

mkdir logs

#Loading modules
num_envs=16
env_name=$1
task_name=$2
seed=$3

exp_name=${env_name}_${task_name}-BSSTORM-1M-seed_${seed}
wandb_exp_name=gitstorm_${env_name}${task_name}_seed${seed}
proj_name=GIT-STORM-Benchmark
# MUJOCO_GL=egl nohup python train_dmc.py  --config-name BSSTORM_DMC  \
MUJOCO_GL=egl python train_dmc.py  --config-name BSSTORM_DMC  \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/BSSTORM_DMC.yaml" \
BasicSettings.env_name="${env_name}" \
BasicSettings.task_name="${task_name}" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
JointTrainAgent.NumEnvs=${num_envs} \
JointTrainAgent.TrainDynamicsEverySteps=${num_envs} \
JointTrainAgent.TrainAgentEverySteps=${num_envs} \
wandb.exp_name=${wandb_exp_name} \
wandb.project_name=${proj_name} 
# > 'logs/'${exp_name}'.out' \
# 2> 'logs/'${exp_name}'.err'


# srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=10:00:00 --pty bash -i

#!/bin/bash


# TODO: if you are running on a cluster, change the info below to match your cluster's configuration.
#SBATCH -J WeatherForcastDMCGITSTORM
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task 18 
#SBATCH --gpus 1 
#SBATCH --mail-type=END
#SBATCH --mail-user=itiszikram@gmail.com
#SBATCH --mem=94G

# TODO: chnage these lines according to your cluster's configuration
module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM

mkdir logs

#Loading modules
num_envs=16
env_name=$1
task_name=$2
seed=$3

exp_name=${env_name}_${task_name}-GITSTORM-life_done-wm_2L512D8H-1M-seed_${seed}
wandb_exp_name=gitstorm_${env_name}${task_name}_seed${seed}
proj_name=GIT-STORM-Benchmark
MUJOCO_GL=egl nohup python train_dmc.py  --config-name GITSTORM_DMC  \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/GITSTORM_DMC.yaml" \
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

srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=10:00:00 --pty bash -i

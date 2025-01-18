#!/bin/bash


# TODO: if you are running on a cluster, change the info below to match your cluster's configuration.
#SBATCH -J Maskgit-STORM
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task 18 
#SBATCH --gpus 1 
#SBATCH --mail-type=END
#SBATCH --mail-user=c.meo@tudelft.nl
#SBATCH --mem=60G

# TODO: chnage these lines according to your cluster's configuration
# module load 2023
# module load Miniconda3/23.5.2-0
# conda init
# source activate STORM

mkdir logs

#Loading modules

env_name=$1
seed=$2
m=$3
T_revise=$4
T_draft=$5
steps=$6
exp_name=${env_name}-GIT-STORM-life_done-wm_2L512D8H-100k-seed_${seed}_m${m}_Trevise${T_revise}_Tdraft${T_draft}_steps${steps}_suV2
nohup python train.py \
BasicSettings.n=${exp_name} \
BasicSettings.Seed=${seed} \
BasicSettings.config_path="config_files/STORM.yaml" \
BasicSettings.env_name="ALE/${env_name}-v5" \
BasicSettings.trajectory_path="D_TRAJ/${env_name}.pkl" \
JointTrainAgent.SampleMaxSteps=${steps} \
Models.MaskGit.M=${m} \
Models.MaskGit.T_revise=${T_revise} \
Models.MaskGit.T_draft=${T_draft} \
> 'logs/'${exp_name}'.out' \
2> 'logs/'${exp_name}'.err'

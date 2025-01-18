
#TODO: change these settings based on your cluster/server
#!/bin/bash
#SBATCH -J Maskgit-STORM-evals
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task 18 
#SBATCH --gpus 1 
#SBATCH --mail-type=END
#SBATCH --mail-user=c.meo@tudelft.nl
#SBATCH --mem=60G


#TODO: change these settings based on your cluster/server
module load 2023
module load Miniconda3/23.5.2-0
# conda init
# source activate STORM
# /home/zikram/GIT-STORM/logs/2024-09-08/02-09-19/ckpt/ckpt/Frostbite-GIT-STORM-life_done-wm_2L512D8H-100k-seed_1_m1_Trevise1_Tdraft1_steps102000_suV2
#Loading modules
# Frostbite-GIT-STORM-life_done-wm_2L512D8H-100k-seed_1_m1_Trevise1_Tdraft1_steps102000_suV2
config_path=config_files/STORM.yaml # TODO: change this path to the path of your config file
env_name=$1 # name of the environment
run_name=$2 # folder name of the run where the checkpoints are saved: example Boxing-life_done-wm_2L512D8H-100k-seed_2_128_embed_dim_128_mlp_dim_128
nohup python eval.py -config_path=${config_path} -env_name='ALE/'${env_name}'-v5' -run_name=${run_name} \
> "logs/"${run_name}_evaluation".out" 2> 'logs/'${run_name}_evaluation'.err'
#!/bin/bash

module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM


# Declare variables 
declare -a All_env_names=(walker walker cheetah)
declare -a All_task_names=(walk run run)
declare -a All_seeds=(0 1 2)
declare -a All_num_layers=(2 3 5)

for i in "${!All_env_names[@]}"; do
    for layers in "${All_num_layers[@]}"; do
        for seed in "${All_seeds[@]}"; do
            env_name=${All_env_names[$i]}
            task_name=${All_task_names[$i]}
            
            sbatch train_dmc_storm.sh $env_name $task_name $seed $layers
        done
    done
done

# sbatch train_dmc_storm.sh walker walk 2 5
# sbatch train_dmc_storm.sh walker walk 1 5



# for env_name in "${All_env_names[@]}"
# do
#     for task_name in "${All_task_names[@]}"
#     do
#         for seed in "${All_seeds[@]}"
#         do
#             sbatch train_dmc_git_storm.sh $env_name $task_name $seed 
#         done
#     done
# done

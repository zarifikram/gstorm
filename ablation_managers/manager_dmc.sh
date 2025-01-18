#!/bin/bash

module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM


# Declare variables 
declare -a All_env_names=(walker)
declare -a All_task_names=(walk)
declare -a All_seeds=(0)

for env_name in "${All_env_names[@]}"
do
    for task_name in "${All_task_names[@]}"
    do
        for seed in "${All_seeds[@]}"
        do
            sbatch train_dmc_storm.sh $env_name $task_name $seed 
        done
    done
done

for env_name in "${All_env_names[@]}"
do
    for task_name in "${All_task_names[@]}"
    do
        for seed in "${All_seeds[@]}"
        do
            sbatch train_dmc_git_storm.sh $env_name $task_name $seed 
        done
    done
done

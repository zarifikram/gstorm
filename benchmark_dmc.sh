#!/bin/bash

module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM


# Declare variables 
declare -a All_env_names=(acrobot cartpole cartpole cartpole cartpole cheetah cup finger finger finger hopper hopper pendulum quadruped quadruped reacher reacher walker walker walker)
declare -a All_task_names=(swingup balance balance_sparse swingup swingup_sparse run catch spin turn_easy turn_hard hop stand swingup run walk easy hard run stand walk)
declare -a All_seeds=(0 1 2 3 4)

# for i in "${!All_env_names[@]}"; do
#     for seed in "${All_seeds[@]}"; do
#         env_name=${All_env_names[$i]}
#         task_name=${All_task_names[$i]}

#         sbatch train_dmc_storm.sh $env_name $task_name $seed
#     done
# done

for i in "${!All_env_names[@]}"; do
    for seed in "${All_seeds[@]}"; do
        env_name=${All_env_names[$i]}
        task_name=${All_task_names[$i]}

        sbatch train_dmc_git_storm.sh $env_name $task_name $seed
    done
done

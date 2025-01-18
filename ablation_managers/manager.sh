#!/bin/bash


module load 2023
module load Miniconda3/23.5.2-0
conda init
source activate STORM


# Declare variables 
declare -a All_vocab_dims=(32 64 128) #this has to be equal with embed_dim
declare -a All_embed_dims=(32 64 128)
declare -a All_mlp_dims=(64 128 256)
declare -a All_env_names=(Boxing Hero MsPacman)
declare -a All_seeds=(0 1 2)



for env_name in "${All_env_names[@]}"
do
    for vocab_dim in "${All_vocab_dims[@]}"
    do
        for seed in "${All_seeds[@]}"
        do
            for mlp_dim in "${All_mlp_dims[@]}"
            do
                sbatch train.sh $env_name $vocab_dim $vocab_dim $mlp_dim $seed
            done
        done
    done
done

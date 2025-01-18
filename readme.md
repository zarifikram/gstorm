# Git-STORM

## Create STORM environment

```
conda create -n STORM python=3.10
conda activate STORM
pip install -r requirements.txt
```

## Running the experiments

Each person has a manager file with his name. The manager file automatically launches the experiments for the corresponding person.

Each person needs to configure ```manager.sh``` and ````train.sh```` with the correct settings for the cluster. Moreover, the path where the checkpoints are saved needs to be updated in ```config_files/STORM.yaml```.

If you are using an editor like VS-CODE, you can search for all the occurrences of ```TODO``` in the project to find the places that need to be updated.

## Evaluation

There is a global manager file ```eval_manager.sh``` that helps launching multiple evaluations at the same time. Similar to training, there are ```TODO``` comments attached to paths or configurations you need to change, depending on your cluster settings/configuration.

You should find relevant ```TODO``` comments in ```eval_manager.sh```, ```evals.sh```, ```eval.py```. 
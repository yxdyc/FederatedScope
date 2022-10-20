set -e

wandb sweep sweep_ditto.yaml
wandb sweep sweep_fedEM.yaml
wandb sweep sweep_fedBN.yaml
wandb sweep sweep_fedrep.yaml
wandb sweep sweep_pFedMe.yaml
wandb sweep sweep_fedAvg_FT.yaml
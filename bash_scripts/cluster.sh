#!/bin/bash
#SBATCH --job-name=refactoring_test                                  # Job name
#SBATCH --time=1-23:00:00                                     # time limit
#SBATCH --cpus-per-task=4                                   # Ask for 4 CPUs
#SBATCH --mem=16Gb                                           # Ask for 16 GB of RAM
#SBATCH --gres=gpu                                          # Ask for GPU
#SBATCH --output=/scratch/axelbogos/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/axelbogos/logs/slurm-%j-%x.error  # log file

# Arguments
# $1: Path to code directory
# Git clone repo
git clone https://github.com/PulkitMadan/convolution-vs-attention.git

# Copy code dir to snapshot code dir (rsync outside of compute node/remember to execute if updated code)
rsync --bwlimit=10mb -av convolution-vs-attention ~/scratch/code-snapshots/ --exclude .git 
rm -r convolution-vs-attention
# Copy snapshot code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
# Copy data to SLURM dir (done once)
cd $SLURM_TMPDIR/"$1/src"

# Setup environment
module purge
module load StdEnv/2020
module load python/3.9.6
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ../requirements_cluster.txt

# Prints
echo "Currently using:"
echo $(which python)
echo "in:"
echo $(pwd)
echo "sbatch file name: $0"
echo $(python -V)
echo $(pip3 show torch)

#fix for convnext model
cp ~/scratch/code-snapshots/convolution-vs-attention/src/utils/helpers.py $SLURM_TMPDIR/venv/lib/python3.9/site-packages/timm/models/layers/helpers.py

# Added Wandb API key
wandb login $WANDB_API_KEY


# Run Script
# training
# --name => Name of your Wandb experiment.
python train.py --train --model resnet --pretrain --name refactored_resnet_test_run --frozen
# testing
python train.py --model resnet --pretrain --load

#Example with $1 -> set 'scratch/code-snapshots/convolution-vs-attention/'
#sbatch convolution-vs-attention/bash_scripts/cluster.sh scratch/code-snapshots/convolution-vs-attention/

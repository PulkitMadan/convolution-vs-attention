#!/bin/bash
#SBATCH --job-name=convatt                                  # Job name
#SBATCH --time=1-23:00:00                                     # time limit
#SBATCH --cpus-per-task=4                                   # Ask for 4 CPUs
#SBATCH --mem=16Gb                                           # Ask for 16 GB of RAM
#SBATCH --gres=gpu                                          # Ask for GPU
#SBATCH --output=/scratch/jizhouw/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/jizhouw/logs/slurm-%j-%x.error  # log file

# Arguments
# $1: Path to code directory
# Copy code dir to snapshot code dir (rsync outside of compute node/remember to execute if updated code)
# rsync --bwlimit=10mb -av convolution-vs-attention ~/scratch/code-snapshots/ --exclude .git 

# Copy snapshot code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
# Copy data to SLURM dir (done once)
# rsync --bwlimit=10mb -av stylized_tiny_imagenet ~/projects/def-sponsor00/datasets/ --exclude .git
cd $SLURM_TMPDIR/"$1/src"

# Setup environment
module purge
module load StdEnv/2020
module load python/3.9.6
#loading opencv module changes python/3.9.6 => python/3.8.10
#module load opencv/4.5.1
#module load pytorch/1.8.1
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
#python -m pip install numpy pandas scikit-learn
python -m pip install -r ../requirements_cluster.txt
#pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Prints
echo "Currently using:"
echo $(which python)
echo "in:"
echo $(pwd)
echo "sbatch file name: $0"
echo $(python -V)
echo $(pip3 show torch)

# Run Script
# training
python train.py --train --model resnet --pretrain --load
# testing
python train.py --model resnet --pretrain --load

#Example with $1
#sbatch convolution-vs-attention/bash_scripts/cluster.sh scratch/code-snapshots/convolution-vs-attention/

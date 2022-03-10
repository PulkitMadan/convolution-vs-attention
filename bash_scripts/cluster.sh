#!/bin/bash
#SBATCH --job-name=convatt                                  # Job name
#SBATCH --time=00:30:00                                     # time limit
#SBATCH --cpus-per-task=1                                   # Ask for 1 CPUs
#SBATCH --mem=2Gb                                           # Ask for 1 GB of RAM
#SBATCH --output=/scratch/jizhouw/logs/slurm-%j-%x.out   # log file
#SBATCH --error=/scratch/jizhouw/logs/slurm-%j-%x.error  # log file

# Arguments
# $1: Path to code directory
# Copy code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
cd $SLURM_TMPDIR/"$1"

# Setup environment
module purge
module load StdEnv/2020
module load python/3.9.6
export PYTHONUNBUFFERED=1
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn
python -m pip install -r requirements.txt
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
python train.py --train data/train.csv --test data/test.csv

#Example
#sbatch convolution-vs-attention/bash_scripts/cluster.sh ~/scratch/code-snapshots/convolution-vs-attention/
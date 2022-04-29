
# Project Title

* [Repo URL](https://github.com/PulkitMadan/convolution-vs-attention)
* [Experiment tracking URL](https://experiment_tracking_url)

A brief description of what this project does and who it's for

## Authors

- [@AxelBogos](https://www.github.com/AxelBogos)
- [@Pulkit Madan](https://www.github.com/PulkitMadan)
- [@Jizhou Wang](https://www.github.com/Jawing)
- [@Abhay Puri](https://www.github.com/abhaypuri)

## Demo

Insert screenshots, gif or link to demo

## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Create virtual environment

* with venv:

```bash
  python3 -m venv /path-to-new-virtual-env/<env_name>
  (or virtualenv /path-to-new-virtual-env/<env_name>)
  source /path-to-new-virtual-env/<env_name>/bin/activate
  pip install -r requirements.txt
```

* with conda:

```bash
conda create --name <env_name> --file requirements.txt
conda activate <env_name>
```

* with conda (environment.yml)

Conda uses the provided `environment.yml` file. You can ignore `requirements.txt` if you choose this method. Make sure
you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [Anaconda](https://www.anaconda.com/products/individual) installed on your system. Once installed, open up your
terminal (or Anaconda prompt if you're on Windows). Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate <env_name>

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=<env_name>

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment
rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment/requirements file using the commands below:

    conda env export > environment.yml
    conda list -e > requirements.txt
    pip list --format=freeze > requirements.txt

Run training, select --model (resnet,vit,convnext,coatnet):

    python train.py --train --model resnet --pretrain --load

Run testing/visualization:

    python train.py --model resnet --pretrain --load

Additional arguments for `train.py`:

* `--frozen`: Freeze model parameters except for the last layer
* `--mela`: Use the melanoma dataset
* `--combined_data`: Use the combined original IN and SIN dataset
* `--name`: Set the name of the run in WandB

## Run on Cluster

In home directory, run the following:

    git clone https://github.com/PulkitMadan/convolution-vs-attention.git
    rsync --bwlimit=10mb -av convolution-vs-attention ~/scratch/code-snapshots/ --exclude .git
    sbatch convolution-vs-attention/bash_scripts/cluster.sh scratch/code-snapshots/convolution-vs-attention/

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`API_KEY`

`ANOTHER_API_KEY`

## WandB Run Locally

Install wandb library and login: <br>
```pip install wandb``` <br>
```wandb login``` <br>
You have to put your API key. You can get it from here https://wandb.ai/authorize

Then you can run your respective Python command.

## Documentation

[Documentation](https://linktodocumentation)


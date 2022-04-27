## Template Instructions and Structure Overview:<br>

### Template Instructions

1. Clone or directly paste the content of this repository inside your project dir of choice
2. Edit this present README with your project detail below
3. If you wish to remove all `info.txt` files in the project once you are familiar with the structure (see below), you
   can run the following script from the root of the project:

```python
import os
import glob

files = glob.glob('./**/info.txt', recursive=True)

for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))
```

4. Finally, remove these instructions and the below structure from this `README.md` file :smile:! Ready to go!

### Structure Overview

```
├── README.md            <- The top-level README for this project.
├── data
│   ├── interim          <- Intermediate data sets that has been transformed, but not yet ready to be used as model input. Basically "Work in progress" data.
│   ├── processed        <- Processed data sets, ready to be used as model input.
│   └── raw              <- Raw, untouched, original data
│
├── docs                 <- An empty dir to be eventually used as a doc dir, eg for `pydoc` or `sphinx`
│
├── bash_scripts         <- Directory for .sh script files
|
├── models             
│   ├── preds            <- Directory for preds of models. Can have sub-directories named after particular models.
│   ├── trained_models   <- Directory for trained and serialized models (eg .pkl, .pt) files
|
├── notebooks            <- Directory for notebooks exploration. Naming convention: ##-initials-description
|                         Example: '06-ab-initial-data-exploration'
│
├── reports              <- Directory for pdf, LaTeX or HTML files related to an eventual report.
│   └── figures          <- Directory for figures to be included in an report (eg. png, jpgs, html plots).
│
├── src                  <- Source code for use in this project.
│   ├── __init__.py      <- Makes src a Python module
│   │ 
│   ├── data             <- Scripts to download or generate data
│   │
│   ├── features         <- Scripts to turn raw data into features for modeling
│   │
│   ├── models           <- Scripts to train models and then use trained models to make predictions
│   │
│   ├── utils            <- Scripts to train models and then use trained models to make predictions
|   │   ├── args.py      <- default argument parser
|   │   ├── defines.py   <- default definitions (eg paths)
│   │
│   └── visualization    <- Scripts to create exploratory and results oriented visualizations
│
├── .gitignore           <- gitignore file, pre-set with common ignores for jupyter notebooks, python and latex
│
├── setup.py             <- Make this project pip installable with `pip install -e`
|
├── requirements.txt     <- The requirements file for reproducing the environment, e.g.
│                         generated with `pip freeze > requirements.txt` or `pip list --format=freeze > requirements.txt` if from a conda env
│   
├── .env                 <- This file is not in this repository as it is by default in `gitignore`. You might need a local version however. 
```

<!---
Don't delete below here!
-->

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


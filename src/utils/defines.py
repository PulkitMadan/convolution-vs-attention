import os
from pathlib import Path

# Define data directory
DATA_DIR = Path(os.path.join('data')).resolve()

# Define dir for storing trained models as .pth files
TRAINED_MODEL_DIR = Path(os.path.join('..', 'models', 'trained_models')).resolve()


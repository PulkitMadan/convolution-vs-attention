import os
from pathlib import Path

# Define data directory
DATA_DIR = Path(os.path.join('data')).resolve()

# Define dir for storing trained models as .pth files
TRAINED_MODEL_DIR = Path(os.path.join('..', 'models', 'trained_models')).resolve()
print(DATA_DIR)
print(TRAINED_MODEL_DIR)

assert DATA_DIR.is_dir(), f'{DATA_DIR} is not a valid path'
assert TRAINED_MODEL_DIR.is_dir(), f'{TRAINED_MODEL_DIR} is not a valid path'

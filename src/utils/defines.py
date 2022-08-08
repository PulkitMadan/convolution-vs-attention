import os
from pathlib import Path

# Define data directory
DATA_DIR = Path(os.path.join(os.path.abspath(__file__),'..','..','..', 'data')).resolve()

# Define dir for storing trained models as .pth files
TRAINED_MODEL_DIR = Path(os.path.join('..', 'models', 'trained_models')).resolve()

print(f'Root data dir: {DATA_DIR}')
print(f'Saving models in: {TRAINED_MODEL_DIR}')

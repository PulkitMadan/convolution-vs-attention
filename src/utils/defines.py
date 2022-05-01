import os
from pathlib import Path

# Define data directory
if os.name == 'nt':  # windows
    DATA_DIR = Path(os.path.join('..', 'data')).resolve()
else:  # linux (cluster)
    home_path = os.path.expanduser('~')
    DATA_DIR = Path(home_path, 'projects', 'def-sponsor00', 'datasets').resolve()

# Define dir for storing trained models as .pth files
TRAINED_MODEL_DIR = Path(os.path.join('..', 'models', 'trained_models')).resolve()

print(f'Root data dir: {DATA_DIR}')
print(f'Saving models in: {TRAINED_MODEL_DIR}')

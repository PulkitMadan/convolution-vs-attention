import os
from pathlib import Path

# Define data directory
DATA_DIR = Path(os.path.join(os.path.abspath(__file__), '..', '..', '..', 'data')).resolve().as_posix()

# Define dir for storing trained models as .pth files
TRAINED_MODEL_DIR = Path(os.path.join(os.path.abspath(__file__), '..', '..', '..', 'trained_models')).resolve().as_posix()

RUNS_OUTPUT_DIR = Path(os.path.join(os.path.abspath(__file__), '..', '..', '..', 'runs_output')).resolve().as_posix()


import os

# Define data directory relative to this file to avoid base PYTHON PATH issues
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# ./output/ by default
output_dir = os.path.join('output')

# .env file path. Assumed to be in root dir
dotenv_file_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')

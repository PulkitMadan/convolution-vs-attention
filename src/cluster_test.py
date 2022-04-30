import os
print(os.getcwd())
print(os.path.abspath(__file__))
from utils import defines

print(defines.TRAINED_MODEL_DIR)
print(defines.DATA_DIR)
model_name = 'VIT16.pth'
home_path = os.path.expanduser('~')
OLD_path_to_model = f'{home_path}/scratch/code-snapshots/convolution-vs-attention/models/trained_models/{model_name}.pth'
NEW_path_to_model = os.path.join(defines.TRAINED_MODEL_DIR, f'{model_name}.pth')

print(f'Old path: {OLD_path_to_model}\nNewpath:{NEW_path_to_model}')

home_path = os.path.expanduser('~')
data_dir = f'{home_path}/projects/def-sponsor00/datasets/stylized_imageNet_subset_OOD'
orig_IN_data_dir = f'{home_path}/projects/def-sponsor00/datasets/ImageNet_subset'

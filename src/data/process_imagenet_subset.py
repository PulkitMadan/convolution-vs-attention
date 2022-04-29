"""
This file processes the subset of ImageNet (consisting of classes defined in `Human16ToTinyImage')
by rename each file by its class id and an id.
"""
import glob
import os
import shutil
from pathlib import Path
from random import sample

from tqdm import tqdm

import Human16ToTinyImage

# Path to the dir with the full ImageNet dataset
DATA_PATH = Path(os.path.abspath(os.sep), 'Volumes', 'FILM ET VHS', 'imagenet-object-localization-challenge', 'ILSVRC',
                 'Data', 'CLS-LOC', 'train').resolve()
OUTPUT_PATH = Path(os.path.abspath(os.sep), 'Users', 'axelbogos', 'IFT6759_datasets', 'IN_subset_final').resolve()
OUTPUT_VAL_DIR = Path(os.path.join(OUTPUT_PATH, 'val')).resolve()
OUTPUT_TEST_DIR = Path(os.path.join(OUTPUT_PATH, 'test')).resolve()
converter = Human16ToTinyImage.ClassConverter()


def get_file_paths(data_path: Path = DATA_PATH, num_train: int = 500, num_val: int = 50, num_test: int = 50):
    num_samples = num_train + num_val + num_test
    len_train = len(str(num_train))
    len_val = len(str(num_val))
    len_test = len(str(num_test))
    for class_dir in tqdm(converter.imgnet_id_to_human16.keys()):
        output_train_dir = os.path.join(OUTPUT_PATH, 'train', class_dir)
        if not os.path.isdir(output_train_dir):
            os.mkdir(output_train_dir)

        files = glob.glob(os.path.join(data_path, class_dir, '*.JPEG'))
        files = sample(files, num_samples)
        train_files = files[0:num_train]
        val_files = files[num_train:num_val + num_train]
        test_files = files[num_val + num_train: num_samples]
        copy_files(train_files, val_files, test_files, len_train, len_val, len_test, output_train_dir)


def copy_files(train_files, val_files, test_files, len_train, len_val, len_test, output_train_dir):
    for idx, file in enumerate(train_files):
        file_name = f"{file.split(os.path.sep)[-1].split('_')[0]}_{str(idx).zfill(len_train)}.JPEG"
        new_file_path = os.path.join(output_train_dir, file_name)
        shutil.copy(file, new_file_path)

    for idx, file in enumerate(val_files):
        file_name = f"{file.split(os.path.sep)[-1].split('_')[0]}_{str(idx).zfill(len_val)}.JPEG"
        new_file_path = os.path.join(OUTPUT_VAL_DIR, file_name)
        shutil.copy(file, new_file_path)

    for idx, file in enumerate(test_files):
        file_name = f"{file.split(os.path.sep)[-1].split('_')[0]}_{str(idx).zfill(len_test)}.JPEG"
        new_file_path = os.path.join(OUTPUT_TEST_DIR, file_name)
        shutil.copy(file, new_file_path)


get_file_paths(DATA_PATH, 500, 50, 50)

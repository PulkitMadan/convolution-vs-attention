"""
This file processes the downsampled 64x64 Imagenet dataset as downloaded from https://image-net.org/download-images.php
The processing consists of :
1. Unpickling the binary files of the train and val batches
2. Filtering the classes to include only classes that have a mapping according to Geirhos 16 Human classes (see 'Human16ToTinyImage.py')
3. Save the individual images as jpgs in directories
"""
import pickle
import glob
from PIL import Image
import os
from pathlib import Path
from src.data import Human16ToTinyImage
import numpy as np

# Path to the dir with the pickled dataset
DATA_PATH = Path(os.path.abspath(os.sep), 'Users', 'axelbogos', 'small_orig_imagenet')
DATA_PATH.resolve()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(file_path):
    d = unpickle(file_path)
    x, y = d['data'], d['labels']
    x = x.reshape((x.shape[0], 64, 64, 3))
    return x, y


def get_file_list(train_file_pattern: str = os.path.join(DATA_PATH, '**', 'train_data_batch_[0-9]'),
                  val_file_pattern: str = os.path.join(DATA_PATH, 'val_data')):
    train_files = glob.glob(train_file_pattern)
    val_files = glob.glob(val_file_pattern)
    return train_files, val_files

def save_images():
    train_files, val_files = get_file_list()
    data, labels = load_databatch(val_files[0])
    geirhos_label_indices = Human16ToTinyImage.ClassConverter().indices_to_label.keys()
    mask = [True if l in geirhos_label_indices else False for l in labels]
    data, labels = data[mask, :], np.array(labels)[mask]
    for i in range(5):
        img = Image.fromarray(data[i])
        img.save(f'test{i}.png')
save_images()



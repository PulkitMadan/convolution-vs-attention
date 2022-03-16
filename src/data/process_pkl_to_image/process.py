"""
This file processes the downsampled 64x64 Imagenet dataset as downloaded from https://image-net.org/download-images.php
The processing consists of :
1. Unpickling the binary files of the train and val batches
2. Filtering the classes to include only classes that have a mapping according to Geirhos 16 Human classes (see 'Human16ToTinyImage.py')
3. Save the individual images as jpgs in directories
"""
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data import Human16ToTinyImage

# Path to the dir with the pickled dataset
DATA_PATH = Path(os.path.abspath(os.sep), 'Users', 'axelbogos', 'small_orig_imagenet').resolve()
OUTPUT_PATH = Path(os.path.abspath(__file__), '..', '..', '..', '..', 'data', 'raw', 'custom_tiny_imagenet').resolve()
converter = Human16ToTinyImage.ClassConverter()
FILE_CACHE = {}


def unpickle(file: str) -> dict:
    """
    Unpickles a file and returns content as a dict
    :param file:
    :return: dict
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(file_path: str, img_size: int = 64) -> tuple:
    """
    Unpickles and loads the data of a batch file as downloaded in https://image-net.org/download-images.php; namely the downsampled 64x64 version
    :param file_path: path to batch file as a string
    :param img_size: Size of the image. Likely 32 or 64 if using downsampled imagenet
    :return: tuple of image data (batch_size x (img_size,img_size,channel)) and labels (batch_size)
    """
    d = unpickle(file_path)
    img_size2 = img_size * img_size
    x, y = d['data'], d['labels']
    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3))
    return x, y


def get_file_list(train_file_pattern: str = os.path.join(DATA_PATH, '**', 'train_data_batch_[0-9]'),
                  val_file_pattern: str = os.path.join(DATA_PATH, 'val_data')) -> tuple:
    """
    Return a list of file paths that match the default name patterns of downloads from Imagenet.
    Example: data_path/somedirs.../train_data_batch0
    :param train_file_pattern: glob style regex for train files
    :param val_file_pattern: glob style regex for val files
    :return: tuple of lists of file paths for train and val files
    """
    train_files = glob.glob(train_file_pattern)
    val_files = glob.glob(val_file_pattern)
    return train_files, val_files


def get_class_sample(file_list: list, class_label: str, sample_size: int) -> tuple:
    """
    Get a sample_sized for 1 class. Opens as many batch_files as necessary to obtain the sample_size of this class.
    Caches every newly opened batch file, hence expect first few calls to be longer than subsequent ones.
    :param file_list: List of batch files
    :param class_label: class of interest
    :param sample_size: size of returned sample
    :return: stacked sample (sample_size x (img_size,img_size,channel) and labels (sample_size)
    """
    stacked_data, stacked_labels = None, None
    class_index = converter.imgnet_id_to_indices[class_label]
    for idx, batch_file in enumerate(file_list):
        if batch_file in FILE_CACHE:
            data, labels = FILE_CACHE[batch_file]
        else:
            FILE_CACHE[batch_file] = load_databatch(batch_file)
            data, labels = FILE_CACHE[batch_file]
        mask = [True if l == class_index else False for l in labels]
        data, labels = data[mask, :], np.array(labels)[mask]
        # Init stacked data accumulation
        if stacked_data is None:
            stacked_data = data
            stacked_labels = labels
        else:
            stacked_data = np.concatenate((stacked_data, data), axis=0)
            stacked_labels = np.concatenate((stacked_labels, labels), axis=0)
        # If stacked is full enough, break
        if stacked_data.shape[0] >= sample_size:
            stacked_data = stacked_data[0:sample_size, :]
            stacked_labels = stacked_labels[0:sample_size]
            break

    if stacked_data.shape[0] < sample_size:
        print(f'Not enough data for class {class_label}')
    return stacked_data, stacked_labels


def save_class_sample(class_counter: int, class_label: str, train_size: int = 500, val_test_ratio: float = 0.1,
                      output_path: str = OUTPUT_PATH) -> None:
    """
    Wrapper function around get_class_sample. Get the sample for a single class and saves it in the dir structure.
    Total sample size is train_size + %ratio necessary for val and test sets.
    :param class_counter: counts the current class being processed. Used for numbering the val and tests images
    :param class_label: current class
    :param train_size: size of train set
    :param val_test_ratio: Ratio of the train set that will be extra data required for val and test.
    :param output_path: path to save images
    :return: None
    """
    assert train_size >= (
            val_test_ratio * 100), f'Sample size should be at least {(val_test_ratio * 100)} image per class'
    assert train_size % (val_test_ratio * 100) == 0, 'Sample size should be divisible by 10'

    # Check if output dirs exists
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(os.path.join(output_path, 'train')):
        os.mkdir(os.path.join(output_path, 'train'))
    if not os.path.isdir(os.path.join(output_path, 'val')):
        os.mkdir(os.path.join(output_path, 'val'))
    if not os.path.isdir(os.path.join(output_path, 'test')):
        os.mkdir(os.path.join(output_path, 'test'))

    train_files, val_files = get_file_list()
    total_files = train_files + val_files
    sample_size = int(train_size + (train_size * 2 * val_test_ratio))
    stacked_data, stacked_labels = get_class_sample(total_files, class_label, sample_size)

    # Save train files
    for idx, img in enumerate(stacked_data[0:train_size, :]):
        if not os.path.isdir(os.path.join(output_path, 'train', class_label)):
            os.mkdir(os.path.join(output_path, 'train', class_label))
        file_path = os.path.join(output_path, 'train', class_label, f'{class_label}_{idx}.JPEG')
        img = Image.fromarray(img)
        img.save(file_path)

    # Save val files
    upper_limit = int((train_size + train_size * val_test_ratio))
    for idx, img in enumerate(stacked_data[train_size:upper_limit, :]):
        file_path = os.path.join(output_path, 'val', f'val_{class_label}_{int(50 * class_counter + idx)}.JPEG')
        img = Image.fromarray(img)
        img.save(file_path)

    # Save test files
    low_limit = int(train_size + train_size * val_test_ratio)
    for idx, img in enumerate(stacked_data[low_limit:, :]):
        file_path = os.path.join(output_path, 'test', f'test_{class_label}_{int(50 * class_counter + idx)}.JPEG')
        img = Image.fromarray(img)
        img.save(file_path)


def save_all_class(train_size: int = 500, val_test_ratio: float = 0.1, output_path: str = OUTPUT_PATH) -> None:
    """
    Wrapper function around save_class_sample. Calls it for every relevent class as defined in Human16ToTinyImage.py
    :param train_size: size of train set
    :param val_test_ratio: Ratio of the train set that will be extra data required for val and test.
    :param output_path: path to save images
    :return: None
    """
    assert train_size >= (
            val_test_ratio * 100), f'Sample size should be at least {(val_test_ratio * 100)} image per class'
    assert train_size % (val_test_ratio * 100) == 0, 'Sample size should be divisible by 10'

    # I sort them to know from where to start back if the process crashes
    labels = list(converter.imgnet_id_to_human16.keys())
    labels.sort()
    class_counter = 0
    for class_label in tqdm(labels, desc='Classes'):
        print(f'\n {class_label}')
        save_class_sample(class_counter, class_label, train_size, val_test_ratio, output_path)
        class_counter += 1


if __name__ == '__main__':
    save_all_class()

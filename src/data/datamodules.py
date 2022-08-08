import os
from typing import List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils import defines
from data.Human16ToTinyImage import ClassConverter
import re
import glob

default_data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ImageBaseDataset(Dataset):
    def __init__(self, img_paths, y, stage, img_transforms=None):
        self.img_paths = img_paths
        self.y = y
        self.stage = stage
        if img_transforms is None:
            self.img_transforms = default_data_transforms
        else:
            self.img_transforms = img_transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        idx = self.img_paths[idx]
        X = Image.open(idx)
        stage_transform = self.img_transforms[self.stage]
        return stage_transform(X), self.y[idx]


class ImageNetDataModule(pl.LightningDataModule):
    """
    Data Module encapsulating the loading and construction of data loaders for the original ImageNet dataset.
    """

    def __init__(self, args=None):
        super().__init__()
        if args:
            self.data_dir = os.path.join(args.data_dir, 'processed', 'imagenet_subset')
            self.batch_size = args.batch_size
        else:
            # Default values for testing
            self.data_dir = os.path.join(defines.DATA_DIR, 'processed', 'imagenet_subset')
            self.batch_size = 16
        self.class_converter = ClassConverter()
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: str = None) -> None:
        """
        This function loads all the images path for the current stage, loads the shape and texture labels
        based on the images names and initializes the BaseDataset for each train, val, test data loaders.
        :param stage: String indicating model stage. Can be "fit" or "test".
        :return: None
        """
        train_imgs_paths = glob.glob(f"{self.data_dir}/train/*/*")
        val_imgs_paths = glob.glob(f"{self.data_dir}/val/*")
        test_imgs_paths = glob.glob(f"{self.data_dir}/test/*")

        if stage == "fit" or stage is None:
            y_train = self._helper_get_labels(train_imgs_paths, r"-|_|\\")
            y_val = self._helper_get_labels(val_imgs_paths, r"-|_")

            self.train = ImageBaseDataset(train_imgs_paths, y_train, 'train')
            self.val = ImageBaseDataset(val_imgs_paths, y_val, 'val')

        if stage == "test" or stage is None:
            y_test = self._helper_get_labels(test_imgs_paths, r"-|_")
            self.test = ImageBaseDataset(test_imgs_paths, y_test, 'test')

    def _helper_get_labels(self, img_paths: List[str], label_regex: str) -> List[List]:
        """
         Iterates an image paths array and extracts the label with a regex. Returns a list of lists as
         [[shape_label,texture_label],...]. Shape and texture label are identical in the original ImageNet.
        :param img_paths: List of image paths
        :param label_regex: regex to extract the label.
        :return:list of lists as [[shape_label,texture_label],...]
        """
        y = list()
        for img in img_paths:
            img_name = img.split('/')[-1]
            label = self.class_converter.imgnet_id_to_indices[re.split(label_regex, img_name)[0]]
            y.append([label, label])
        return y

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class StylizedImageNetDataModule(pl.LightningDataModule):
    """
    Data Module encapsulating the loading and construction of data loaders for the stylized ImageNet dataset.
    """

    def __init__(self, args=None):
        super().__init__()
        if args:
            self.data_dir = os.path.join(args.data_dir, 'processed', 'stylized_imagenet_subset')
            self.batch_size = args.batch_size
        else:
            # Default values for testing
            self.data_dir = os.path.join(defines.DATA_DIR, 'processed', 'stylized_imagenet_subset')
            self.batch_size = 16
        self.class_converter = ClassConverter()
        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage: str = None):
        """
        This function loads all the images path for the current stage, loads the shape and texture labels
        based on the images names and initializes the BaseDataset for each train, val, test data loaders.
        :param stage: String indicating model stage. Can be "fit" or "test".
        :return: None
        """
        train_imgs_paths = glob.glob(f"{self.data_dir}/train/*/*")
        val_imgs_paths = glob.glob(f"{self.data_dir}/val/*")
        test_imgs_paths = glob.glob(f"{self.data_dir}/test/*")

        if stage == "fit" or stage is None:
            y_train = self._helper_get_labels(train_imgs_paths, r"-|_|\\")
            self.train = ImageBaseDataset(train_imgs_paths, y_train, 'train')

            y_val = self._helper_get_labels(val_imgs_paths, r"-|_")
            self.val = ImageBaseDataset(val_imgs_paths, y_val, 'val')

        if stage == "test" or stage is None:
            y_test = self._helper_get_labels(test_imgs_paths, r"-|_")
            self.test = ImageBaseDataset(test_imgs_paths, y_test, 'test')

    def _helper_get_labels(self, img_paths, label_regex):
        """
         Iterates an image paths array and extracts the label with a regex. Returns a list of lists as
         [[shape_label,texture_label],...].
        :param img_paths: List of image paths
        :param label_regex: regex to extract the label.
        :return:list of lists as [[shape_label,texture_label],...]
        """
        y = list()
        for img in img_paths:
            img_name = img.split('/')[-1]
            shape_label = self.class_converter.imgnet_id_to_indices[re.split(label_regex, img_name)[0]]
            texture_label = self.class_converter.imgnet_id_to_indices[re.split(label_regex, img_name)[3]]
            y.append([shape_label, texture_label])
        return y

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

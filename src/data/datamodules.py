import numpy as np
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
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

from data.Human16ToTinyImage import ClassConverter

classconv = ClassConverter()
mapping_207 = dict()
mapping_207_reverse = list(classconv.imgnet_id_to_human16.keys())
for idx, key in enumerate(mapping_207_reverse):
    mapping_207[key] = idx
class_16_listed = list(classconv.human16_to_imgnet_id)


def map207_to_16(num):
    return class_16_listed.index(classconv.imgnet_id_to_human16[num])


def map207_to_16names(num):
    return classconv.imgnet_id_to_human16[num]


class BaseDataset(Dataset):
    def __init__(self, X, y, stage, img_transforms=None):
        self.X = X
        self.y = y
        self.stage = stage
        if img_transforms is None:
            self.img_transforms = default_data_transforms
        else:
            self.img_transforms = img_transforms

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        stage_transform = self.img_transforms[self.stage]
        return stage_transform(self.X[idx]), self.y[idx]


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.data_dir = os.path.join(args.data_dir, 'stylized_imagenet_subset')

    def prepare_data(self) -> None:
        train_imgs_paths = glob.glob(f"{self.data_dir}/train/*/*")
        val_imgs_paths = glob.glob(f"{self.data_dir}/val/*")
        test_imgs_paths = glob.glob(f"{self.data_dir}/test/*")

        self.X_train, self.y_train = self._helper_load_data(train_imgs_paths, r"-|_|\\", 'train')

    def _helper_load_data(self, img_paths, label_regex, stage):
        X = [Image.open(img_path) for img_path in img_paths]
        y = list()
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            shape_texture_label = mapping_207[re.split(label_regex, img_name)[0]]
            if stage == 'train':
                y.append(shape_texture_label)
            elif stage == 'val':
                y.append(shape_texture_label)
            elif stage == 'test':
                y.append(shape_texture_label)
                y.append(shape_texture_label)
        return X, y

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = BaseDataset(self.X_train, self.y_train)
            self.val = BaseDataset(self.X_val, self.y_val)
        if stage == "test" or stage is None:
            self.test = BaseDataset(self.X_test, self.y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

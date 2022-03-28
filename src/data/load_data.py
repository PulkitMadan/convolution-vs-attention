#imports
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import seed_worker
import os 
from PIL import Image
import re

#path to the extracted data
#data directory
home_path = os.path.expanduser('~')
data_dir = f'{home_path}/projects/def-sponsor00/datasets/stylized_imageNet_subset'

from data.Human16ToTinyImage import ClassConverter
classconv = ClassConverter()
#set mapping
mapping_207 = dict()
mapping_207_reverse = list(classconv.imgnet_id_to_human16.keys())
for idx, key in enumerate(mapping_207_reverse):
    mapping_207[key]=idx

class_16_listed = list(classconv.human16_to_imgnet_id)
#class_names = image_datasets_train.classes

#set data transforms
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
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

#custom dataset
class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        #print(image_paths)
        self.transform = transform
        
    def get_class_label(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[0]]) #switch to 1 and 4 for tinyimagenet
        y.append(mapping_207[image_name_split[3]])
        return y
    def get_class_label_train(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_|\\", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[0]])
        y.append(mapping_207[image_name_split[3]])
        return y
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #print(image_path)
        #print(image_path.split('/')[-1])
        x = Image.open(image_path)
        if 'train' in image_path:
            y = self.get_class_label_train(image_path.split('/')[-1])
        else:
            y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)


def dataload(batch_size,data_transforms=data_transforms):

    image_datasets = {x: MyDataset(glob.glob(f"{data_dir}/{x}/*"), data_transforms[x])
                      for x in ['test', 'val']}

    image_datasets['train'] = MyDataset(glob.glob(f"{data_dir}/train/*/*"), data_transforms['train'])
    #print(image_datasets['val'][0][1])
    #print(image_datasets['train'][0][1])
    # image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'),
    #                                           data_transforms['train'])
    
    #num_workers = 0 otherwise dataloader won't process fast
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4, worker_init_fn=seed_worker)
                  for x in ['train','val']}
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                 shuffle=False, num_workers=4)

    # dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=4,
    #                                               shuffle=True, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['test','train','val']}

    return image_datasets,dataloaders,dataset_sizes



def map207_to_16(num):
    return class_16_listed.index(classconv.imgnet_id_to_human16[num])
def map207_to_16names(num):
    return classconv.imgnet_id_to_human16[num]


def print_datamap():
    # print(classconv.human16_to_imgnet_id['knife'])
    # print(mapping_207)
    # print(len(mapping_207_reverse))
    image_datasets,dataloaders,dataset_sizes= dataload(batch_size=64)
    print(dataset_sizes)
    print(dataset_sizes['test'])
    print(class_16_listed)
    print(image_datasets)
    print(image_datasets['train'][0][1])
    print(image_datasets['val'][0][1])

    num = mapping_207_reverse[200]
    print(map207_to_16(num))
    print(map207_to_16names(num))
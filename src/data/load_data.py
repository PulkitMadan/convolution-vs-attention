#imports
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import seed_worker
import os 
from PIL import Image
import re
import pandas as pd

#path to the extracted data
#data directory

#loading stylized_imagenet
if os.name == 'nt': #windows
    data_dir = os.path.abspath(f'../data/processed/Stylized_ImageNet_subset_OOD')
else: #linux
    home_path = os.path.expanduser('~')
    data_dir = f'{home_path}/projects/def-sponsor00/datasets/stylized_imageNet_subset_OOD'

#loading melanoma dataset
if os.name == 'nt': #windows
    data_dir_m = os.path.abspath(f'../data/processed/Melanoma_dataset')
else: #linux
    home_path = os.path.expanduser('~')
    data_dir_m = f'{home_path}/projects/def-sponsor00/datasets/Melanoma_dataset'



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
#linux format
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
    def get_class_label_val(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[0]]) #switch to 1 and 4 for tinyimagenet
        #y.append(mapping_207[image_name_split[3]])
        return y
    def get_class_label_train(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_|\\", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[0]])
        #for commented out for OOD train
        #y.append(mapping_207[image_name_split[3]])
        return y
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #print(image_path)
        #print(image_path.split('/')[-1])
        x = Image.open(image_path)
        if 'train' in image_path:
            y = self.get_class_label_train(image_path.split('/')[-1])
        elif 'val' in image_path:
            y = self.get_class_label_val(image_path.split('/')[-1])
        else:
            y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)

#windows local format
class MyDataset2(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def get_class_label(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_|\\", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[1]])
        y.append(mapping_207[image_name_split[4]])
        return y

    def get_class_label_val(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_|\\", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[1]])
        #y.append(mapping_207[image_name_split[4]])
        return y

    def get_class_label_train(self, image_name):
        # your method here
        image_name_split = re.split(r"-|_|\\", image_name)
        #print(image_name_split)
        y = list()
        #original class then texture class
        y.append(mapping_207[image_name_split[2]])
        #y.append(mapping_207[image_name_split[5]])
        return y
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        #print(image_path)
        #print(image_path.split('/')[-1])
        x = Image.open(image_path)
        if 'train' in image_path:
            y = self.get_class_label_train(image_path.split('/')[-1])
        elif 'val' in image_path:
            y = self.get_class_label_val(image_path.split('/')[-1])
        else:
            y = self.get_class_label(image_path.split('/')[-1])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.image_paths)

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transforms = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        
    def __getitem__(self, index):
        im_path = os.path.join(self.root_dir, self.df.iloc[index]['image_id'] + '.jpg')
        x = Image.open(im_path)
    
        y = self.df.iloc[index]['melanoma']

        if self.transforms:
            x = self.transforms(x)
        
        return x, y
    
    def __len__(self):
        return len(self.df)
    
def dataload_Mela(batch_size,data_transforms=data_transforms):
    image_datasets = {}
    image_datasets['test'] = MelanomaDataset(csv_file=f'{data_dir_m}/ISIC-2017_Test_v2_Part3_GroundTruth.csv',
                                root_dir=f'{data_dir_m}/test',transforms=data_transforms['test'])
    image_datasets['val'] = MelanomaDataset(csv_file=f'{data_dir_m}/ISIC-2017_Validation_Part3_GroundTruth.csv',
                                root_dir=f'{data_dir_m}/val',transforms=data_transforms['val'])
    image_datasets['train'] = MelanomaDataset(csv_file=f'{data_dir_m}/ISIC-2017_Training_Part3_GroundTruth.csv',
                                root_dir=f'{data_dir_m}/train',transforms=data_transforms['train'])

    # print(image_datasets['test'][0])
    # print(image_datasets['val'][0])
    # print(image_datasets['train'][0])
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=4, worker_init_fn=seed_worker)
                  for x in ['train','val']}
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,
                                                 shuffle=False, num_workers=4)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['test','train','val']}

    return image_datasets,dataloaders,dataset_sizes

def dataload(batch_size,data_transforms=data_transforms):
    if os.name == 'nt': #windows
        print('in windows')
        image_datasets = {x: MyDataset2(glob.glob(f"{data_dir}/{x}/*"), data_transforms[x])
                          for x in ['test', 'val']}

        image_datasets['train'] = MyDataset2(glob.glob(f"{data_dir}/train/*/*"), data_transforms['train'])
    else: #linux
        print('in linux')
        image_datasets = {x: MyDataset(glob.glob(f"{data_dir}/{x}/*"), data_transforms[x])
                          for x in ['test', 'val']}

        image_datasets['train'] = MyDataset(glob.glob(f"{data_dir}/train/*/*"), data_transforms['train'])


    # print(image_datasets['val'][0][1])
    # print(image_datasets['train'][0][1])
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
    image_datasets,dataloaders,dataset_sizes= dataload_Mela(batch_size=64)
    print(dataset_sizes)
    # print(dataset_sizes['test'])
    # print(class_16_listed)
    # print(image_datasets)
    print(len(dataloaders['test']))
    print(len(dataloaders['train']))
    print(image_datasets['train'][0][0])
    print(image_datasets['train'][0][1])
    print(image_datasets['val'][0][1])
    print(image_datasets['test'][0][1])


    # image_datasets,dataloaders,dataset_sizes= dataload(batch_size=64)
    # print(dataset_sizes)
    # # print(dataset_sizes['test'])
    # # print(class_16_listed)
    # # print(image_datasets)
    # print(len(dataloaders['test']))
    # print(len(dataloaders['train']))
    # print(image_datasets['train'][0][0])
    # print(image_datasets['train'][0][1])
    # print(image_datasets['val'][0][1])
    # print(image_datasets['test'][0][1])

    num = mapping_207_reverse[200]
    print(map207_to_16(num))
    print(map207_to_16names(num))
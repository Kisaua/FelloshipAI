from warnings import catch_warnings
import pandas as pd
import os
import time
import copy
import torch
import numpy as np
import json
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader 
from torchvision import datasets, models, transforms

THIS_FOLDER = os.path.abspath('')
my_file = os.path.join(THIS_FOLDER, 'oxford-102-flowers/test.txt')
root_dir = os.path.join(THIS_FOLDER, 'oxford-102-flowers/')

class FlowersDataset(Dataset):
    """Flowers dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.flowers = pd.read_csv(csv_file, delimiter = " ")
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.flowers)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.flowers.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.flowers.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample


data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

with open(os.path.join(THIS_FOLDER, 'oxford-102-flowers/cat_to_name.json')) as json_file:
    cat_to_name = json.load(json_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_datasets = {x : FlowersDataset(csv_file= os.path.join(THIS_FOLDER, 'oxford-102-flowers/',x+'.txt'), root_dir=root_dir, 
                                transform = data_transforms[x]) for x in ['train', 'valid', 'test']}

dataloaders = {x : DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=0) for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
fig = plt.figure()
for i in range(len(image_datasets)):
    sample = image_datasets['train'][i]

    #print(i, sample['image'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    label = str(sample['label'])
    ax.set_title('#{}'.format(cat_to_name[label]))
    ax.axis('off')
    img = sample['image']
    img = img.permute(1, 2, 0)
    #print (img)
    plt.imshow(img)
    #plt.imshow()
    if i == 3:
        plt.show()
        break

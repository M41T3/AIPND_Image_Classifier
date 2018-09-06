#major functions for predict and train
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from datetime import datetime

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='flowers', help='Directory of a dataset. Must include /dir/train/ ,           /dir/valid/ and /dir/test/.')

    return parser.parse_args()

def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    print(train_dir)
    #  Define your transforms for the training, validation, and testing sets
    data_transforms = {"train": transforms.Compose([transforms.Resize(255), transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),  
                                                transforms.CenterCrop(224), transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "test": transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "valid": transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                      "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
                      "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=4, shuffle=True),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=4, shuffle=True),
                   "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=4, shuffle=True)}
    
    class_to_idx = image_datasets["train"].class_to_dict
    
    return dataloaders, class_to_idx

if __name__ == "__main__":
    print("Collection of functions to create, train and test models.")
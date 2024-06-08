


from random import random
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import torch

import numpy as np
import os
from collections import Counter
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Grayscale, Resize, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.io import read_image
import matplotlib.pyplot as plt
from utils import *

from torchvision.models import vit_b_32, vit_b_16, swin_t

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TrainingAndTestingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.training = datasets.ImageFolder(os.path.join(root_dir, 'training'), transform=transform)
        self.testing = datasets.ImageFolder(os.path.join(root_dir, 'testing'), transform=transform)

    def __len__(self):
        return len(self.training) + len(self.testing) 

    def __getitem__(self, index):
        if index < len(self.training_dataset):
            image = self.training_dataset[index]
            return image
        else:
            fixed_index = index - len(self.training_dataset)
            return self.testing_dataset[fixed_index]

class MixDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
            self.dataset_dir = dataset_dir
            self.transform = transform
            self.classes = os.listdir(dataset_dir)
            self.num_classes = len(self.classes)

            self.data = []

            for class_name in self.classes:
                class_path = os.path.join(dataset_dir, class_name)
                for filename in os.listdir(class_path):
                    img_path = os.path.join(class_path, filename)
                    label = self.classes.index(class_name)
                    self.data.append((img_path, label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def iterate_dataset(self):
        index = 0

        while index < self.__len__():
            yield self.__getitem__(index)
            index += 1

    def get_sample_each_class(self):
        
        image_class = set()
        index = 0

        while index < self.__len__():
            img, label = self.__getitem__(index) 

            if label not in image_class:
                yield img, label  
                image_class.add(label) 

            index += 1 

def draw_each_class(dataset:Dataset):

    for img,label in dataset.get_sample_each_class():

        plt.imshow(img.permute((1, 2, 0)), cmap='gray')  # Ajusta las dimensiones de la imagen si es necesario
        plt.title(f'Class {label}')
        plt.axis('off') 
        plt.show()      

def show_samples(X, Y):
    counter =1

    for img,label in zip(X[:9],Y[:9]):
        plt.subplot(3,3,i)
        plt.imshow(img.permute((1,2,0)), cmap='gray')
        plt.title(label.item())
        plt.xticks([])
        plt.yticks([])
        counter += 1

def convert_seconds(time_seconds):
    """
    Convert a given number of seconds to hours, minutes, and seconds.
    Params:
        integer: The total number of seconds to be converted.
    Returns:
        str: A string representing the converted time in the format "{hours} h, {minutes} m, {seconds} s".
    """
    hours = time_seconds // 3600
    minutes = (time_seconds % 3600) // 60
    seconds = time_seconds % 60
    return f"{hours} h, {minutes} m, {seconds} s"
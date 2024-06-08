
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

from torchvision.models import __dict__ as models_dict
from torchvision.models import *
import torchvision.models as torch_models

import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import time
from abc import ABC, abstractmethod

from torchinfo import summary
#from torchsummary import summary
import numpy
    
import torch.nn as nn
import torchvision

class BaseModel(nn.Module):

    def __init__(self, model_name, **kwargs):
         
        super(BaseModel, self).__init__()

        weights = kwargs.get('weights', None)
        transfer_learning = kwargs.get('transfer_learning', False)

        if hasattr(torch_models, model_name):

            self.model_instance = getattr(torch_models, model_name)(weights=weights)
            self.model_class = model_name
            print(f'Model class: {self.model_class}. Weights: {weights}')
            
        else:
            self.model_class = f'No module found: {model_name} in torchvision.models \n\nList of Models: {list(models_dict.keys())}'
            raise ValueError(self.model_class)
        
        if transfer_learning:
            self.transfer_learning(**kwargs)
        
        self.build_timestamp = int(time.time() * 1000)
        self.MODELS_DIR = f'{self.model_class}_{self.build_timestamp}'

    def transfer_learning(self, **kwargs):

        num_classes = kwargs.get('num_classes', None)
        dropout = kwargs.get('dropout', 0.0)
        transfer_learning = kwargs.get('transfer_learning', False)
        
        model_modifier = FactoryMethod().create_modifier(self.model_instance)
        model_modifier.create_classifier(num_classes, dropout)

        if transfer_learning:
            model_modifier.freeze_all_layer()
            model_modifier.active_classifier_layer()
            
    def train(self, dataloader, loss_fn, optimizer, losses, accuracy, batch_size, **kwargs):
        
        device = kwargs.get('device', 'cpu')

        validationlaoder = kwargs.get('validation_dataloader', None)
        val_acc = kwargs.get('val_acc', None)
        val_loss = kwargs.get('val_loss', None)

        size = len(dataloader.dataset)
        n=size//(batch_size*10)

        self.model_instance.to(device)
        self.model_instance.train()

        correct = 0
        validation_acc, validation_loss, best_val_acc = 0, 0, 0
        results_val_acc, results_val_loss = [], []

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = self.model_instance(X)
            train_loss = loss_fn(pred, y)

            # Backpropagation
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % n == 0:
                current = (batch + 1) * len(X)
                train_acc = 100*correct / size            

                if validationlaoder:
                    validation_acc, validation_loss = self.evaluate(validationlaoder, loss_fn, device = device)
                    val_info = f"--- Validation Accuracy: {validation_acc:>0.1f}%, Validation loss: {validation_loss:>8f}"
                    results_val_acc.append(validation_acc)
                    results_val_loss.append(validation_loss)
                    best_val_acc = max(best_val_acc, validation_acc)

                print(f"Training Accuracy: {train_acc:>0.1f}%, Training loss: {train_loss.item():>8f} {val_info if 'val_info' in locals() else ''} [{current:>5d}/{size:>5d}] Steps completed:[{(current / size * 100):>3.0f} %] ")                    

        train_acc = 100*correct/size
        accuracy.append(train_acc)
        losses.append(train_loss.item())

        mean_val_acc = numpy.mean(results_val_acc)
        mean_val_loss = numpy.mean(results_val_loss)

        if validationlaoder is not None:
            val_acc.append(mean_val_acc)
            val_loss.append(mean_val_loss)
            val_info = f"--- Validation Average Accuracy: {mean_val_acc:>0.1f}%, Validation Average loss: {mean_val_loss:>8f}"

        print(f"Training Accuracy: {train_acc:>0.1f}%, Training Average loss: {train_loss.item():>8f}  {val_info if validationlaoder else ''} [{size:>5d}/{size:>5d}] Steps completed:[{(size / size * 100):>3.0f} %]")

        return max(results_val_acc), results_val_loss[ results_val_acc.index(max(results_val_acc)) ], best_val_acc
    
    def evaluate(self, dataloader, loss_fn, losses = None, accuracy = None, device = 'cuda'):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model_instance.eval()
        loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model_instance(X)
                loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss /= num_batches
        acc = 100*correct/size
        losses.append(loss) if losses != None else None
        accuracy.append(acc) if accuracy != None else None
        return acc, loss

    def save_model(self):
        torch.save(self.model_instance.state_dict(), f'saved_models/{self.get_model_class()}_{self.build_timestamp}.pth')
        print(f"Model saved to saved_models/{self.get_model_class()}_{self.build_timestamp}.pth")

    def load_state_model(self, dir):
        torch.load(dir)
        print(f"Model state loaded from {dir}")

    def draw_graphics(self, train_loss, train_acc, val_acc, val_loss):

        plt.figure()
        plt.plot(train_loss, color='blue')
        plt.plot(val_loss, color='red')
        plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(f'graphics/{self.get_model_class()}_avg_loss_{self.build_timestamp}.svg')
        
        plt.figure()
        plt.plot(train_acc, color='blue')
        plt.plot(val_acc, color='red')
        plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(f'graphics/{self.get_model_class()}_Accuracy_{self.build_timestamp}.svg')


    def get_model(self):
        return self.model_instance
    
    def get_model_class(self):
        return self.model_class


class ClassifierModifier(ABC):

    def __init__(self, model_instance):
        self.model_instance = model_instance
        
    def createLinear(self, in_features, num_classes, dropout):

        linear = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(
                    in_features=in_features,
                    out_features=num_classes,
                    bias=True)
            ).requires_grad_(True)
            
        return linear

    def create_classifier(self, num_classes, dropout):
        vector_features = self.get_in_features()      
        new_classifier = self.createLinear(vector_features, num_classes, dropout)    
        self.set_head(new_classifier)

    def freeze_all_layer(self):
        for parameter in self.model_instance.parameters():
            parameter.requires_grad = False

    @abstractmethod
    def get_in_features(self):
        pass
    
    @abstractmethod
    def set_head(self):
        pass

    @abstractmethod
    def active_classifier_layer(self):
        pass 

class FactoryMethod:

    @staticmethod
    def create_modifier(instance):
        
        if isinstance(instance, torchvision.models.maxvit.MaxVit):
            return MaxVitModifier(instance)
    
        if isinstance(instance, torchvision.models.swin_transformer.SwinTransformer):
            return SwinModifier(instance)
        
        if isinstance(instance, torchvision.models.vision_transformer.VisionTransformer):
            return VitModifier(instance)
        else:
            raise ValueError("Unsupported model type")

class MaxVitModifier(ClassifierModifier):

    def set_head(self, new_classifier):
        self.model_instance.classifier[-1] = new_classifier   

    def get_in_features(self):
        in_features = self.model_instance.classifier[-1].in_features
        return in_features

    def active_classifier_layer(self):
        for parameter in self.model_instance.classifier[-1].parameters():
            parameter.requires_grad = True

class SwinModifier(ClassifierModifier):

    def set_head(self, new_classifier):
        self.model_instance.head = new_classifier   

    def get_in_features(self):
        in_features = self.model_instance.head.in_features
        return in_features
    
    def active_classifier_layer(self):
        for parameter in self.model_instance.head.parameters():
            parameter.requires_grad = True

                   
class VitModifier(ClassifierModifier):
  
    def set_head(self, new_classifier):
        self.model_instance.heads = new_classifier   

    def get_in_features(self):
        in_features = self.model_instance.heads
        in_features = in_features.head.in_features
        return in_features
    
    def active_classifier_layer(self):
        for parameter in self.model_instance.heads.parameters():
            parameter.requires_grad = True


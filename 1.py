import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from pathlib import Path
from lenet5 import LeNet5
from tqdm import tqdm
import os



def get_transform():
    if train:

        return transforms.Compose({
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32,32)),

            trans

        })
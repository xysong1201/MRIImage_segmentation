
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms, utils
import numpy as np
from PIL import Image
from random import randint
import time


# In[2]:


gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')


# In[30]:


class SimpleNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SimpleNet, self).__init__()
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(128),
            nn.ReLU(),
             nn.Conv2d(128, 256, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(256),
            nn.ReLU(),
             nn.Conv2d(256, 512, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(512),
            nn.ReLU(),
             nn.Conv2d(512, 1024, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, out_channel, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self,x):
        x = self.doubleconv(x)
        x = F.log_softmax(x, dim =1)
        return x


# In[31]:



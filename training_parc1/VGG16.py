
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
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
print (device)


# In[3]:


class VGG16(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            # 1 x 128 x 128 -> 64 x 64 x64
            nn.Conv2d(in_channel, 64, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            # 64 x 64 x 64 -> 128 x 32 x 32
            nn.Conv2d(64, 128, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # 128 x 32 x 32 -> 256 x 16 x 16
            nn.Conv2d(128, 256, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            
            # 256 x 16 x16 -> 512 x 8 x 8
            nn.Conv2d(256, 512, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
            #   bs x 512, 8, 8 -> 163840 -> 4096 -> 4096 -> 178
        self.layer2 = nn.Sequential( 
            
            nn.Linear(32768,4096),
            nn.ReLU(),
       
            nn.Linear(4096,4096),
            nn.ReLU(),
            
            nn.Linear(4096,out_channel)
        )
        
        
    def forward(self,x):
        x = self.layer1(x)
        x = x.view(-1, 32768)
        x = self.layer2(x)
        x = F.log_softmax(x, dim =1)
        return x


# In[4]:


# model =VGG16(1,178)


# In[5]:


# print(model)


# In[6]:


# nb_param=0
# for param in model.parameters():
#     nb_param+=np.prod(list(param.data.size()))
# print(nb_param)


# In[7]:


# x=torch.rand(5,1,128,128)
# y = model(x)
# print(y.size())



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




class DenseBlock(nn.Module):
    '''
    param:
    in_channel, out_channel
    '''

    def __init__(self, in_channel, out_channel):
        super(DenseBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 5, padding = 2 ),
            nn.BatchNorm2d(out_channel),
            nn.PReLU()          
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channel+in_channel, out_channel, kernel_size = 5, padding = 2 ),
            nn.BatchNorm2d(out_channel),
            nn.PReLU() 
        )
        
        self.outconv = nn.Conv2d(2*out_channel+in_channel, out_channel, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.block1(x)
        x2 = torch.cat((x, x1), dim=1)
        x3 = self.block2(x2)
        x4 = torch.cat((x,x1,x3), dim=1)
        
        out = self.outconv(x4)
        
        return out


# In[4]:


class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices=True)
        self.conv = DenseBlock(in_channel, out_channel)     
                   
    def forward(self, x):
        x1, indices = self.maxpool(x)
        x1 = self.conv(x1)
        return x1, indices


# In[5]:


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BottleNeck, self).__init__()
        
        self.pool = nn.MaxPool2d(2, stride = 2, return_indices=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size = 5, padding = 2)
        self.BatchNorm = nn.BatchNorm2d(out_channel)
       
    def forward(self,x):
        x , indices = self.pool(x)
        x = self.conv(x)
        x = self.BatchNorm(x)
        return x, indices


# In[6]:


class up(nn.Module):
    def __init__(self, in_channel, out_channel ):
        super(up, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv = DenseBlock(in_channel*2, out_channel)
      
    
    def forward(self, x1, x2, indice = None, output_size=None):
        x1 = self.unpool(x1, indice, output_size)
        x = torch.cat([x1,x2], dim = 1)
        x = self.conv(x)
        
        return x


# In[7]:


class OutConv(nn.Module):
    def __init__(self, in_channel, out_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_classes, 1)
    def forward(self,x):
        x = self.conv(x)
        return x


# In[8]:


class Encoder(nn.Module):
    def __init__(self, in_channel, num_channel):
        super(Encoder, self).__init__()
        self.conv_in = DenseBlock(in_channel, num_channel)
        
        self.down1 = down(num_channel, num_channel)
        self.down2 = down(num_channel, num_channel)
        self.down3 = down(num_channel, num_channel) 
        
        self.BottleNeck = BottleNeck(num_channel, num_channel)
    def forward(self, x):
        x1 = self.conv_in(x)
        
        x2, idx1 = self.down1(x1)
        x3, idx2 = self.down2(x2)
        x4, idx3 = self.down3(x3)
        x5, idx4 = self.BottleNeck(x4)
        
        return x5, x4, x3, x2, x1, idx1, idx2, idx3, idx4


# In[9]:


class Decoder(nn.Module):
    def __init__(self, in_channel, out_class):
        super(Decoder, self).__init__()
        
        self.up1 = up(in_channel, in_channel)
        self.up2 = up(in_channel, in_channel)
        self.up3 = up(in_channel, in_channel)
        self.up4 = up(in_channel, in_channel)
        self.conv_out = OutConv(in_channel, out_class)

    
    def forward(self,x, idx1, idx2, idx3, idx4, x4,x3,x2,x1):
        x = self.up1.forward(x,x4,idx4)
        x = self.up2.forward(x,x3,idx3)
        x = self.up3.forward(x,x2,idx2)
        x = self.up4.forward(x,x1,idx1)
        x = self.conv_out(x)
        x = F.log_softmax(x, dim =1)
        return x       


# In[10]:


class QuickNAT(nn.Module):
    def __init__(self, in_channel, num_channel, out_class):
        super(QuickNAT, self).__init__()
        self.in_channel = in_channel
        self.num_channel = num_channel
        self.out_class = out_class
        self.encoder = Encoder(self.in_channel, self.num_channel)
        self.decoder = Decoder(self.num_channel, self.out_class)
        
    def forward(self,x):
        x5, x4, x3, x2, x1, idx1, idx2, idx3, idx4 = self.encoder(x)
        x = self.decoder.forward(x5, idx1, idx2, idx3, idx4, x4,x3,x2,x1)
        return x


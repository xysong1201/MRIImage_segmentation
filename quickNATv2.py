
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
from enum import Enum

import torch.nn.functional as F


# In[2]:


class DenseBlock(nn.Module):
    '''
    param:
    in_channel, out_channel, drop_out
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


# In[3]:


class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices=True)
        self.conv = DenseBlock(in_channel, out_channel)     
                   
    def forward(self, x):
        x1, indices = self.maxpool(x)
        x1 = self.conv(x1)
        return x1, indices
    


# In[4]:


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


# In[5]:


class up(nn.Module):
    def __init__(self, in_channel, out_channel, indice = None, output_size=None):
        super(up, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv = DenseBlock(in_channel*2, out_channel)
        self.indice = indice
        self.output_size = output_size
    
    def forward(self, x1, x2):
        x1 = self.unpool(x1,self.indice, self.output_size)
        x = torch.cat([x1,x2], dim = 1)
        x = self.conv(x)
        
        return x
        


# In[6]:


class OutConv(nn.Module):
    def __init__(self, in_channel, out_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_classes, 1)
    def forward(self,x):
        x = self.conv(x)
        return x
    


# In[7]:


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
           
    


# In[8]:


class Decoder(nn.Module):
    def __init__(self, in_channel, out_class, idx1, idx2, idx3, idx4, x4,x3,x2,x1):
        super(Decoder, self).__init__()
        
        self.up1 = up(in_channel, in_channel, indice = idx4, output_size=x4.size())
        self.up2 = up(in_channel, in_channel, indice = idx3, output_size=x3.size())
        self.up3 = up(in_channel, in_channel, indice = idx2, output_size=x2.size())
        self.up4 = up(in_channel, in_channel, indice = idx1,output_size=x1.size())     
        
        self.conv_out = OutConv(in_channel, out_class)
        self.x4 = x4
        self.x3 = x3
        self.x2 = x2
        self.x1 = x1
    
    def forward(self,x):
        x = self.up1(x,self.x4)
        x = self.up2(x,self.x3)
        x = self.up3(x,self.x2)
        x = self.up4(x,self.x1)
        x = self.conv_out(x)
        x = F.log_softmax(x, dim =1)
        return x
        


# In[9]:


class QuickNAT(nn.Module):
    def __init__(self, in_channel, num_channel, out_class):
        super(QuickNAT, self).__init__()
        self.encoder = Encoder(in_channel, num_channel)
        self.num_channel = num_channel
        self.out_class = out_class
    def forward(self,x):
        x5, x4, x3, x2, x1, idx1, idx2, idx3, idx4 = self.encoder(x)
        decoder = Decoder(self.num_channel, self.out_class, idx1, idx2, idx3, idx4, x4,x3,x2,x1)
        x = decoder(x5)
        return x


# In[10]:





# coding: utf-8

# In[12]:


import torch
import os
import torch.nn as nn
import torch.nn.functional as F


# #### Define the 3D Unet

# In[14]:


class double_conv3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(double_conv3d, self).__init__()
        
        self.doubleconv3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.doubleconv3d(x)
        return x


# In[15]:


class down3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down3d, self).__init__()
        self.downconv3d = nn.Sequential(
            nn.MaxPool3d(2, stride = 2),
            double_conv3d(in_channel, out_channel)
        )
    def forward(self, x):
        x = self.downconv3d(x)
        return x 


# In[16]:


class up3d(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(up3d, self).__init__()
        self.up3d = nn.ConvTranspose3d(in_channel, out_channel, kernel_size = 2, stride = 2)
        self.conv3d = double_conv3d(in_channel, out_channel)
    def forward(self, x1, x2):
        # after the convtranspose2d, the output W,H doubled.
        x1 = self.up3d(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # crop x2 
        x2 = x2[:,:,diffY//2:diffY//2+x1.size()[2] , diffX//2:diffY//2+x1.size()[3]] 
        x = torch.cat([x2,x1], dim = 1)
        x = self.conv3d(x)
        return x


# In[17]:


class outconv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv3d(x)
        return x


# In[18]:


class UNet3D(nn.Module):
    """
    Args:
        input channel(int)
        output channel(int)
    """
    def __init__(self):
        super(UNet3D, self).__init__()
        self.conv_in = double_conv3d(3,8)

        self.down1 = down3d(8,16)
        self.down2 = down3d(16,32)
        self.down3 = down3d(32,64)
        self.down4 = down3d(64,128)
        
        self.up1 = up3d(128,64)
        self.up2 = up3d(64,32)
        self.up3 = up3d(32,16)
        self.up4 = up3d(16,8)
        
        self.conv_out = outconv3d(8,3)
       

    def forward(self, x):
        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)
        x = F.log_softmax(x, dim =1)
        return x


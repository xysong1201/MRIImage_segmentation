import torch.nn.functional as F
import torch.nn as nn
import torch


class double_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(double_conv, self).__init__()
        
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, padding =1), #same padding
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size = 3,padding =1), #same padding
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.doubleconv(x)
        return x





class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.downconv = nn.Sequential(
            nn.MaxPool2d(2, stride = 2),
            double_conv(in_channel, out_channel)
        )
    def forward(self, x):
        x = self.downconv(x)
        return x        





class up(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size = 2, stride = 2)
        self.conv = double_conv(in_channel, out_channel)
    def forward(self, x1, x2):
        # after the convtranspose2d, the output W,H doubled.
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]  
        diffX = x2.size()[3] - x1.size()[3]
        # crop x2 
        x2 = x2[:,:,diffY//2:diffY//2+x1.size()[2] , diffX//2:diffX//2+x1.size()[3]] 
        x = torch.cat([x2,x1], dim = 1)
        x = self.conv(x)
        return x


# In[7]:


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# In[8]:


class UNet(nn.Module):
    """
    Args:
        input channel(int)
        output channel(int)
    """
    def __init__(self, in_channel,out_class,num_channel):
        super(UNet, self).__init__()      
        
        self.conv_in = double_conv(in_channel,num_channel)
        self.down1 = down(num_channel,num_channel*2)
        self.down2 = down(num_channel*2,num_channel*4)
        self.down3 = down(num_channel*4,num_channel*8)
        self.down4 = down(num_channel*8,num_channel*16)
        
        self.up1 = up(num_channel*16,num_channel*8)
        self.up2 = up(num_channel*8,num_channel*4)
        self.up3 = up(num_channel*4,num_channel*2)
        self.up4 = up(num_channel*2,num_channel)
        self.conv_out = outconv(num_channel,out_class)
        
       
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



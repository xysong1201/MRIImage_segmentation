
# coding: utf-8

# In[1]:


import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset,  WeightedRandomSampler
from skimage import io, transform
from torchvision import transforms, utils
import torch
import numpy as np
import nibabel as nib
from random import randint
from PIL import Image
import torch.optim as optim
import time
import QuickNAT as QN
import torch.nn as nn
import sys


sys.setrecursionlimit(10000)

# ### The dataset of T1w image and brainmask image in axial plane

# In[2]:


class TrainDataset(Dataset):
    """Training dataset with mask image mapping to classes"""
    def __init__(self, T1a_dir, brainmask_dir, transform=None):
        """
        Args:
            T1a_dir (string): Directory with T1w image in axial plane
            transform (callable): Optional transform to be applied on a sample
            brainmask_dir (string): Directory with brainmask in axial plane
        """
        self.T1a_dir = T1a_dir
        self.transform = transform
        self.brainmask_dir = brainmask_dir
#         self.mapping = {
#             180:91
#         }
        
#     def mask_to_class(self, mask):
#         for k in self.mapping:
#             mask[mask==k] = self.mapping[k]
#         return mask
    
    def __len__(self):
        T1a_list = os.listdir(self.T1a_dir)
        return len(T1a_list)
    
    
    def __getitem__(self, idx):
        T1a_list = os.listdir(self.T1a_dir)
        brainmask_list = os.listdir(self.brainmask_dir)
        
        T1a_str = T1a_list[idx]
        
        T1a_arr = io.imread(os.path.join(self.T1a_dir, T1a_str))
        T1a_tensor = torch.from_numpy(T1a_arr)
       
        
        compose_T1 = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((128,128),interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        T1a_tensor = torch.unsqueeze(T1a_tensor, dim = 0)   

        T1a_tensor = compose_T1(T1a_tensor)
       
        
        # The original brainmask value is 0,1,2     
        brainmask_str = brainmask_list[idx]
    
        brainmask_arr = io.imread(os.path.join(self.brainmask_dir, brainmask_str))
        brainmask_tensor = torch.from_numpy(brainmask_arr)
        
        compose = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((128,128),interpolation=Image.NEAREST), 
                                      transforms.ToTensor()])
        
        brainmask_tensor = torch.unsqueeze(brainmask_tensor, dim = 0)
        brainmask_tensor = compose(brainmask_tensor)
        brainmask_tensor = brainmask_tensor.squeeze()
        
     
        # After the resize, the value of brainmask is 0, 0.0039 and 0.0078, so this formula below is used
        # to make it to 0, 1, 2
        brainmask_tensor = torch.round(brainmask_tensor / 0.0039).byte()   
        
#         parc1a_tensor = self.mask_to_class(parc1a_tensor)
      
        sample = {'T1a':T1a_tensor, 'brainmask':brainmask_tensor}
        
        if self.transform:
            T1a = self.transform(T1a_tensor)
            sample = {'T1a':T1a, 'brainmask':brainmask}
            
        return sample



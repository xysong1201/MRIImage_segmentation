
# coding: utf-8

# In[2]:


import os
from torch.utils.data import Dataset, DataLoader
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


# In[3]:


class TrainDataset(Dataset):
    """Training dataset with mask image mapping to classes"""
    def __init__(self, T1a_dir, parc1a_dir, transform=None):
        """
        Args:
            T1a_dir (string): Directory with T1w image in axial plane
            transform (callable): Optional transform to be applied on a sample
            parc1a_dir (string): Directory with parcellation scale 5 in axial plane
        """
        self._T1a_dir = T1a_dir
        self.transform = transform
        self._parc1a_dir = parc1a_dir
        self.mapping = {
            180:91
        }
        
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask
    
    def __len__(self):
        T1a_list = os.listdir(self._T1a_dir)
        return len(T1a_list)
    
    
    def __getitem__(self, idx):
        T1a_list = os.listdir(self._T1a_dir)
        parc1a_list = os.listdir(self._parc1a_dir)
        
        T1a_str = T1a_list[idx]
        
        T1a_arr = io.imread(os.path.join(self._T1a_dir, T1a_str))
        T1a_tensor = torch.from_numpy(T1a_arr)
        
        compose_T1 = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize((128,128),interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        T1a_tensor = torch.unsqueeze(T1a_tensor, dim = 0)
        T1a_tensor = compose_T1(T1a_tensor)
              
        parc1a_str = parc1a_list[idx]
    
        parc1a_arr = io.imread(os.path.join(self._parc1a_dir, parc1a_str))
        parc1a_tensor = torch.from_numpy(parc1a_arr)
        
        compose = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((128,128),interpolation=Image.NEAREST), 
                                      transforms.ToTensor()])
        
        parc1a_tensor = torch.unsqueeze(parc1a_tensor, dim = 0)
        parc1a_tensor = compose(parc1a_tensor)
        parc1a_tensor = parc1a_tensor.squeeze()
        
        parc1a_tensor = torch.round(parc1a_tensor / 0.0039).byte()
        parc1a_tensor = self.mask_to_class(parc1a_tensor)
      
        sample = {'T1a':T1a_tensor, 'parc1a':parc1a_tensor}
        
        if self.transform:
            T1a = self.transform(T1a_tensor)
            sample = {'T1a':T1a, 'parc1a':parc1a}
            
        return sample


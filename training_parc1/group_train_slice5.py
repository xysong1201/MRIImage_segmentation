
# coding: utf-8

# In[1]:


import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
from my_Dataset import TrainDataset
import sys



sys.setrecursionlimit(10000)


# In[2]:


gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')
print(device)


# In[3]:


sub_idx = 0
slice_idx = 5
T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
parc1a_dir = '/home/xiaoyu/MRIdata_group/parc_1/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
total_data_0 = TrainDataset(T1a_dir=T1a_dir, parc1a_dir = parc1a_dir)


# In[4]:


for sub_idx in range(1,330):
    T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
    parc1a_dir = '/home/xiaoyu/MRIdata_group/parc_1/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
    train_data = TrainDataset(T1a_dir=T1a_dir, parc1a_dir = parc1a_dir)
    total_data_0 = total_data_0 + train_data
print(len(total_data_0))


# In[5]:


dataloader_0 = DataLoader(total_data_0, batch_size = 4, shuffle = True, num_workers = 4)
print(len(dataloader_0))


# In[6]:


start=time.time()
model = QN.QuickNAT(1,128,178)
nb_param=0
for param in model.parameters():
    nb_param+=np.prod(list(param.data.size()))
print(nb_param)
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters() ,lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# In[7]:
print('Slice group ID: ', slice_idx)

for epoch in range(0,720):
    running_loss = 0
    num_batches = 0
    scheduler.step()
    for i_batch, sample_batched in enumerate(dataloader_0):
    
        optimizer.zero_grad()
        
        #get the inputs
        inputs, labels = sample_batched['T1a'], sample_batched['parc1a']
        

        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs.requires_grad_()
        
        #forward + backward +optimize
        scores = model(inputs)

          
        # Define the loss
        loss = criterion(scores, labels.long()) 
        loss.backward()
        optimizer.step()
        
        # compute and accumulate stats
        running_loss += loss.detach().item()

       
        num_batches+=1 
        
        
    # AVERAGE STATS THEN DISPLAY    
    total_loss = running_loss/num_batches
   
    elapsed = (time.time()-start)/60
        
    print('epoch=',epoch, '\t time=', elapsed,'min', '\t loss=', total_loss ) 
 
print('Finish Training')




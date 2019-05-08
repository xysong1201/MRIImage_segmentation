
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
from QN import QuickNat
import torch.nn as nn
from my_Dataset import TrainDataset
from weight_axial import weight
import sys



# In[2]:
sys.setrecursionlimit(10000)

gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')
print(device)


# In[3]:


sub_idx = 0
slice_idx = 3
T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
parc1a_dir = '/home/xiaoyu/MRIdata_group/parc_1/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
total_data = TrainDataset(T1a_dir=T1a_dir, parc1a_dir = parc1a_dir)


# In[4]:


for sub_idx in range(1,330):
    T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
    parc1a_dir = '/home/xiaoyu/MRIdata_group/parc_1/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
    train_data = TrainDataset(T1a_dir=T1a_dir, parc1a_dir = parc1a_dir)
    total_data = total_data + train_data
print(len(total_data))


print('The slice id is:',slice_idx)

dataloader = DataLoader(total_data, batch_size = 1, shuffle = True, num_workers = 4)
print(len(dataloader))


# In[ ]:


start=time.time()
params= {'num_channels':1, 'num_filters':64, 'kernel_h':5, 'kernel_w':5, 'kernel_c':1, 'stride_conv':1,'pool':2, 
         'stride_pool':2, 'num_class':178, 'se_block': 'CSSE','drop_out':0.2}
model = QuickNat(params)
nb_param=0
for param in model.parameters():
    nb_param+=np.prod(list(param.data.size()))
print(nb_param)



model = model.to(device)
weight = weight.to(device)
criterion = nn.NLLLoss(weight = weight)
optimizer = optim.Adam(model.parameters() ,lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)



loss_list = []
for epoch in range(0,2000):
    running_loss = 0
    num_batches = 0
    scheduler.step()
    for i_batch, sample_batched in enumerate(dataloader):
    
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
    loss_list = np.append(loss_list,total_loss)
   
    elapsed = (time.time()-start)/60
        
    print('epoch=',epoch, '\t time=', elapsed,'min', '\t loss=', total_loss ) 
    if epoch%100==0:
        print(loss_list)
 
print('Finish Training')






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
import torch.nn as nn
from my_Dataset import TrainDataset
# import matplotlib.pyplot as plt
from weight_axial import weight,count
# import res18 as Res
from QN import QuickNat

import sys




sys.setrecursionlimit(10000)


# In[2]:


gpu_id = 2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')
print(device)


# In[3]:


sub_idx = 0
slice_idx = 5
T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
brainmask_dir = '/home/xiaoyu/MRIdata_group/brainmask/axial/sub{}/group{}'.format(sub_idx,slice_idx)
total_data = TrainDataset(T1a_dir=T1a_dir, brainmask_dir = brainmask_dir)


# In[4]:


for sub_idx in range(1,330):
    T1a_dir = '/home/xiaoyu/MRIdata_group/T1w/axial/sub{}/slice{}'.format(sub_idx,slice_idx)
    brainmask_dir = '/home/xiaoyu/MRIdata_group/brainmask/axial/sub{}/group{}'.format(sub_idx,slice_idx)
    train_data = TrainDataset(T1a_dir=T1a_dir, brainmask_dir = brainmask_dir)
    total_data = total_data + train_data
print(len(total_data))


# In[ ]:


dataloader = DataLoader(total_data, batch_size = 4, shuffle = True, num_workers = 4)
print(len(dataloader))


# In[ ]:


# total_data[0]['T1a'].shape


# In[ ]:


params= {'num_channels':1, 'num_filters':64, 'kernel_h':5, 'kernel_w':5, 'kernel_c':1, 'stride_conv':1,'pool':2, 
         'stride_pool':2, 'num_class':3, 'se_block': 'CSSE','drop_out':0.2}
model = QuickNat(params)
# model = Res.res101()
start=time.time()
nb_param=0
for param in model.parameters():
    nb_param+=np.prod(list(param.data.size()))
print(nb_param)


# In[ ]:


# print(model)


# In[ ]:


bs=5
x=torch.rand(bs,1,128,128)
y = model(x)
print(y.size())


# In[ ]:


model = model.to(device)
weight = weight.to(device)
criterion = nn.NLLLoss(weight = weight)
optimizer = optim.Adam(model.parameters() ,lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.85)


# In[ ]:


loss_list = []
for epoch in range(0,1000):
    running_loss = 0
    num_batches = 0
    scheduler.step()
    for i_batch, sample_batched in enumerate(dataloader):
    
        optimizer.zero_grad()
        
        #get the inputs
        inputs, labels = sample_batched['T1a'], sample_batched['brainmask']
        

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
    if epoch%10==0:
        print(loss_list)
        torch.save(model.state_dict(),'model')
   
 
print('Finish Training')


# In[ ]:


# plt.plot(loss_list)


# In[ ]:


# l = loss_list[0]
# sum_dif = []
# for loss in loss_list:
#     dif = l-loss
#     l = loss
# #     num = np.random.randint(1, 20)
# #     print(num)

# #     if(num == 7):
# #         break
#     sum_dif = np.append(sum_dif,dif)
# print(sum_dif)
# plt.plot(sum_dif)


# In[ ]:


# def show_mask(image, mask):
#     """
#     Show image and mask
#     Args:
#         image(numpyarray): The training image
#         semantic(numpyarray): The training image segmentation
#     """    
#     plt.subplot(1,2,1)
#     plt.title('image')
#     plt.imshow(image)
#     plt.subplot(1,2,2)
#     plt.title('mask')
#     plt.imshow(mask)
#     plt.show()


# In[ ]:


# sample=total_data[4]
# img = sample['T1a']
# mask = sample['brainmask']

# show_mask(img.squeeze(), mask)

# img = img.unsqueeze(dim = 0)

# img = img.to(device)


# # feed it to network
# scores =  model(img)
# scores = scores.detach().cpu().squeeze().permute(1,2,0)

# scores = torch.exp(scores)

# a,b = torch.max(scores,dim=2)
# plt.imshow(b)


# In[ ]:





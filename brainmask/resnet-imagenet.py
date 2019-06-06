
# coding: utf-8

# In[1]:


import torch
import numpy as np
import os.path
import utils


# In[ ]:


from utils import check_imagenet_dataset_exists


# In[ ]:


data_path=check_imagenet_dataset_exists()

train_data=torch.load(data_path+'imagenet/train_data.pt')
train_label=torch.load(data_path+'imagenet/train_label.pt')
test_data=torch.load(data_path+'imagenet/test_data.pt')
test_label=torch.load(data_path+'imagenet/test_label.pt')

print(train_data.size())
print(test_data.size())


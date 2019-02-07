
# coding: utf-8

# In[1]:


import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
import utils_xy
from torchvision import transforms, utils
import torch
import numpy as np
import nibabel as nib
from random import randint
from PIL import Image
import torch.optim as optim
import time
import models.UNet as UNet
import torch.nn as nn


# #### Set the visible GPU

# In[2]:


gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')
print (device)


# We have 330 subjects, for every subject, there is 3 planes for T1w image, and 3 planes for Parc_5 image.
# Pick a random subject

# # I. Data Loading and preprocessing

# In[3]:


for sub_idx in range(330):
    T1a_dir = '/home/xiaoyu/MRIdata/T1w/axial/sub{}'.format(sub_idx)
    T1s_dir = '/home/xiaoyu/MRIdata/T1w/sagittal/sub{}'.format(sub_idx)
    T1c_dir = '/home/xiaoyu/MRIdata/T1w/coronal/sub{}'.format(sub_idx)
    
    parc5a_dir = '/home/xiaoyu/MRIdata/parc_5/axial/sub{}'.format(sub_idx)
    parc5s_dir = '/home/xiaoyu/MRIdata/parc_5/sagittal/sub{}'.format(sub_idx)
    parc5c_dir = '/home/xiaoyu/MRIdata/parc_5/coronal/sub{}'.format(sub_idx)
    
    T1a_list = os.listdir(T1a_dir)
    T1s_list = os.listdir(T1s_dir)
    T1c_list = os.listdir(T1c_dir)
    
    parc5a_list = os.listdir(parc5a_dir)
    parc5s_list = os.listdir(parc5s_dir)
    parc5c_list = os.listdir(parc5c_dir)
    
    if sub_idx == 0: # set sub0 as test set.
        print('\nT1w Axial slices num:',len(T1a_list))
        print('T1w Sagittal slices num:',len(T1s_list))
        print('T1w Coronal slices num:',len(T1c_list))

        print('\nParc5 Axial slices num:',len(parc5a_list))
        print('Parc5 Sagittal slices num:',len(parc5s_list))
        print('Parc5 Coronal slices num:',len(parc5c_list))
        continue
    

    
    


# In[4]:


# axial plane
for i in range(len(T1a_list)):
    T1a_dir = '/home/xiaoyu/MRIdata/T1w/axial/sub{}'.format(sub_idx)
    parc5a_dir = '/home/xiaoyu/MRIdata/parc_5/axial/sub{}'.format(sub_idx)
    T1a_str = T1a_list[i]
    parc5a_str = parc5a_list[i]
    T1a_arr = io.imread(os.path.join(T1a_dir, T1a_str))
    parc5a_arr = io.imread(os.path.join(parc5a_dir, parc5a_str))
    
    print('Image array data type: ', T1a_arr.dtype)
    print('Mask array data type: ', parc5a_arr.dtype)
    
    # make 2 tensors from the numpy array for image and mask respectively
    T1a_tensor = torch.from_numpy(T1a_arr)
    parc5a_tensor = torch.from_numpy(parc5a_arr)
    
    # observe the data size of the image and mask
    print('\nImage data size: ', T1a_tensor.size())
    print('Mask data size:', parc5a_tensor.size())
    
    if i ==0:
        break    


# #### Define a funciton for the visualization

# In[5]:


def show_mask(image, mask):
    """
    Show image and mask
    Args:
        image(numpyarray): The training image
        semantic(numpyarray): The training image segmentation
    """    
    plt.subplot(1,2,1)
    plt.title('image')
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.title('mask')
    plt.imshow(mask)
    plt.show()
    
# test the function
show_mask(T1a_arr, parc5a_arr)


# ### Find the unique color in mask

# In[6]:


colors = torch.tensor([])
for i in range(len(T1a_list)):
    
    parc5a_dir = '/home/xiaoyu/MRIdata/parc_5/axial/sub{}'.format(sub_idx)
   
    parc5a_str = parc5a_list[i]
   
    parc5a_arr = io.imread(os.path.join(parc5a_dir, parc5a_str))
    
    parc5a_tensor = torch.from_numpy(parc5a_arr)
    
    unique_color = torch.unique(parc5a_tensor).type(torch.FloatTensor)
    colors = torch.cat((colors,unique_color))
colors = torch.unique(colors)
print(colors)
sorted_color, indices = torch.sort(colors)
print(sorted_color)
print(colors.dtype)
print(colors.size())


# Found that there are 256 unique color in the mask, so there are 256 classes. There is no need to map since the class label is [0,255]

# #### Define the Training set Class

# In[7]:


class TrainDataset(Dataset):
    """Training dataset with mask image mapping to classes"""
    def __init__(self, T1a_dir, parc5a_dir, transform=None):
        """
        Args:
            T1a_dir (string): Directory with T1w image in axial plane
            transform (callable): Optional transform to be applied on a sample
            parc5a_dir (string): Directory with parcellation scale 5 in axial plane
        """
        self.T1a_dir = T1a_dir
        self.transform = transform
        self.parc5a_dir = parc5a_dir
        
    def __len__(self):
        T1a_list = os.listdir(self.T1a_dir)
        return len(T1a_list)
    
    
    def __getitem__(self, idx):
        T1a_list = os.listdir(T1a_dir)
        parc5a_list = os.listdir(parc5a_dir)
        
        T1a_str = T1a_list[idx]
        
        T1a_arr = io.imread(os.path.join(T1a_dir, T1a_str))
        T1a_tensor = torch.from_numpy(T1a_arr)
        
        compose_T1 = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize((128,128),interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        T1a_tensor = torch.unsqueeze(T1a_tensor, dim = 0)
        T1a_tensor = compose_T1(T1a_tensor)
              
        parc5a_str = parc5a_list[idx]
    
        parc5a_arr = io.imread(os.path.join(parc5a_dir, parc5a_str))
        parc5a_tensor = torch.from_numpy(parc5a_arr)
        
        compose = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize((128,128),interpolation=Image.NEAREST), 
                                      transforms.ToTensor()])
        
        parc5a_tensor = torch.unsqueeze(parc5a_tensor, dim = 0)
        parc5a_tensor = compose(parc5a_tensor)
        parc5a_tensor = parc5a_tensor.squeeze()
        
        parc5a_tensor = torch.round(parc5a_tensor / 0.0039).byte()
      
        sample = {'T1a':T1a_tensor, 'parc5a':parc5a_tensor}
        
        if self.transform:
            T1a = self.transform(T1a_tensor)
            sample = {'T1a':T1a, 'parc5a':parc5a}
            
        return sample


# In[8]:


train_data = TrainDataset(T1a_dir=T1a_dir, parc5a_dir = parc5a_dir)
print('Total image number: {}'.format(len(train_data)))
colors = torch.tensor([])

for i in range(len(train_data)):
    sample = train_data[i]
    mask = sample['parc5a']
    unique_color = torch.unique(mask).float()
    colors = torch.cat((colors,unique_color))
colors = torch.unique(colors)
print(colors)
sorted_color, indices = torch.sort(colors)
print(sorted_color)


# ### Test the traindataset class 
# We can see here 
# * T1wa data is in the range [-1,1], the data type is torch.float32 
# * parc5a data value is (0,1,2,3...255), data type is torch.uint8  

# In[9]:


train_data = TrainDataset(T1a_dir=T1a_dir, parc5a_dir = parc5a_dir)
print('Total T1a image number: {}'.format(len(train_data)))

maximum = torch.tensor([0],dtype=torch.float32)
minimum = torch.tensor([0],dtype=torch.float32)

for i in range(len(train_data)):
    sample = train_data[i]
    T1a = sample['T1a']
    parc5a = sample['parc5a']
        
    maxi = torch.max(T1a)
    mini = torch.min(T1a)

    if maximum < maxi:
        maximum = torch.max(T1a)
    if minimum > mini:
        minimum = torch.min(T1a)
 
print(maximum)
print(minimum)


# In[10]:


for i in range(len(train_data)):
    sample = train_data[i]
    T1a = sample['T1a']
    parc5a = sample['parc5a']
    
    print('T1a info:')
    print(T1a.size())
    print(T1a.dtype)
    print(torch.max(T1a))
    print(torch.min(T1a))
    
    print('\n parc5a info:')
    print(parc5a.dtype)
    print(parc5a.size())
    print(torch.max(parc5a))
    print(torch.min(parc5a))
    print(type(T1a))
    print('\nVisualization:')
    show_mask(T1a.squeeze(), parc5a)
    if i == 2:  
        break


# ### Use the dataloader in Pytorch to form the train dataset
# Data loader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

# In[11]:


dataloader = DataLoader(train_data, batch_size = 5, shuffle = True, num_workers = 4)


# In[12]:


print(len(dataloader))


# In[13]:


for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched['T1a'].size())
    print(sample_batched['parc5a'].size())
    if i_batch ==0:
        break


# ### Unet parameters

# In[14]:


# input channel is 3, output channel is 1
unet = UNet.UNet(1,256,64)
print(unet)
unet_params = list(unet.parameters())
print("The length of the unet parameter is: ")
print(len(unet_params))
print("The conv1's weight: ")
print(unet_params[0].size()) # conv1's weight  0.4.
print("The weight's dtype: ")
print(unet_params[0].dtype)

nb_param=0
for param in unet.parameters():
    nb_param+=np.prod(list(param.data.size()))
print(nb_param)


# In[15]:


bs=5
x=torch.rand(bs,1,128,128)
y = unet(x)
print(y.size())


# ### Make sure the current device

# In[16]:


current_device = torch.cuda.current_device()
torch.cuda.device(current_device)


# In[17]:


torch.cuda.device_count()


# In[18]:


torch.cuda.get_device_name(0)


# In[ ]:


unet = unet.to(device)


# In[ ]:


print(unet_params[0].dtype)


# ### Define the loss function and learning rate

# In[ ]:


criterion = nn.NLLLoss()


# #### Inputs size: bs x 1 x 182 x 217 tensor, which is in range [-1,1], data type is float32
# #### Labels size: bs x 182 x 217 tensor, the values is 0,1,2,... 255, data type is uint8

# In[ ]:


optimizer = optim.Adam(unet.parameters() ,lr=0.001)


start=time.time()
for epoch in range(1,500):
   
    
    # define the running loss
    running_loss = 0
    running_error = 0
    num_batches=0
      
    for i_batch, sample_batched in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        #get the inputs
        inputs, labels = sample_batched['T1a'], sample_batched['parc5a']
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs.requires_grad_()
        
        #forward + backward +optimize
        scores = unet(inputs)
      
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


# #### Choose image at random from the training set and see how good/bad are the predictions

# In[ ]:


T1a_dir = '/home/xiaoyu/MRIdata/T1w/axial/sub20'
parc5a_dir = '/home/xiaoyu/MRIdata/parc_5/axial/sub20'

test_data = TrainDataset(T1a_dir=T1a_dir, parc5a_dir = parc5a_dir)


# In[ ]:


sample=train_data[6]
img = sample['T1a']
mask = sample['parc5a']

show_mask(img.squeeze(), mask)

img = img.unsqueeze(dim = 0)

img = img.to(device)


# feed it to network
scores =  unet(img)
scores = scores.detach().cpu().squeeze().permute(1,2,0)
print(scores.size())
scores = torch.exp(scores)
print(torch.max(scores))
print(torch.min(scores))
a,b = torch.max(scores,dim=2)
plt.imshow(b)


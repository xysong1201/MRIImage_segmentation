
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# In[ ]:


path = '~/'
print(path)


# In[ ]:


trainset = torchvision.datasets.ImageNet(root=path + '/ImageNet/temp', train=True,
                                                download=True, transform=transforms.ToTensor())


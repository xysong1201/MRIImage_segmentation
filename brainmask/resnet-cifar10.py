import torch
import numpy as np
import os.path
import utils
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from utils import check_cifar_dataset_exists
data_path=check_cifar_dataset_exists()

train_data=torch.load(data_path+'cifar/train_data.pt')
train_label=torch.load(data_path+'cifar/train_label.pt')
test_data=torch.load(data_path+'cifar/test_data.pt')
test_label=torch.load(data_path+'cifar/test_label.pt')

print(train_data.size())
print(train_label.size())
print(test_data.size())


# ### Compute average pixel intensity over all training set and all channels

# In[3]:


mean= train_data.mean()

print(mean)


# ### Compute standard deviation

# In[4]:


std= train_data.std()

print(std)


# ### Make a Resnet convnet Class

# In[ ]:


# The BasicBlock is the repeated block in Resnet.
class BasicBlock(nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        # block 1 :  channel x 32 x 32 -> channel x 32 x 32 -> channel x 32 x 32 (2 layers)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels,in_channels, kernel_size=3, stride = 1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # identity shorcut
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))  


# In[ ]:


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        
        # income conv 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        
        self.conv2 = BasicBlock(16)
        self.conv2_1 = BasicBlock(16)
        self.conv2_2 = BasicBlock(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1) # subsampling using a stride of 2.
        self.conv4 = BasicBlock(32)
        self.conv4_1 = BasicBlock(32)
        self.conv4_2 = BasicBlock(32)
        
        self.conv5 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1) # subsampling
        self.conv6 = BasicBlock(64)
        self.conv6_1 = BasicBlock(64)
        self.conv6_2 = BasicBlock(64)
        
#         self.avg_pool = nn.AdaptiveAvgPool2d((8, 8))

        # linear layers:   64 x 8 x 8 --> 4096 --> 10
        self.fc = nn.Linear(4096, 10)
        

    def forward(self, x):
        output = self.conv1(x)
        
        output = self.conv2(output)
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        
        output = self.conv3(output)
        
        output = self.conv4(output)
        output = self.conv4_1(output)
        output = self.conv4_2(output)
        
        output = self.conv5(output)
        
        output = self.conv6(output)
        output = self.conv6_1(output)
        output = self.conv6_2(output)
        output = output.view(output.size(0), -1)
        # bs x 4096 ->  bs * 10
#         print(output.size())

        x = self.fc(output)
        x = F.log_softmax(x, dim =1)

        return x 


# In[ ]:


model = ResNet()
# print(model)


# In[ ]:


utils.display_num_param(model)


# In[ ]:


bs=5
x=torch.rand(bs,3,32,32)
y = model(x)
print(y.size())


# ### Put the network to GPU

# In[ ]:


gpu_id = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')
print(device)


# ### Send the weights of the networks to the GPU (as well as the mean and std)

# In[ ]:


model = model.to(device)

mean=mean.to(device)

std=std.to(device)


# ### Choose the criterion, learning rate, and batch size.

# In[ ]:


criterion = nn.NLLLoss()

my_lr=0.1 

bs= 128


# ### Divide the data to 45k train data and 5k dev data

# In[ ]:


dev_data = train_data[45000:50000]
dev_label = train_label[45000:50000]


# In[ ]:


train_data = train_data[0:45000]
train_label = train_label[0:45000]


# In[ ]:


print(dev_data.size())
print(dev_label.size())


# ### Function to evaluate the network on the test set

# In[ ]:


def eval_on_dev_set():

    running_error=0
    num_batches=0

    for i in range(0,5000,bs):

        minibatch_data =  dev_data[i:i+bs]
        minibatch_label = dev_label[i:i+bs]

        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        inputs = (minibatch_data - mean)/std

        scores= model( inputs ) 

        error = utils.get_error( scores , minibatch_label)

        running_error += error.item()

        num_batches+=1

    total_error = running_error/num_batches
    print( 'error rate on dev set =', total_error*100 ,'percent')


# ### Do 64k passes through the training set. Divide the learning rate by 10 at epoch 32k and 48k

# In[ ]:


start=time.time()

for epoch in range(1,64000):
    
    # divide the learning rate by 10 at epoch 32k and 48k
    if epoch==32000 or epoch == 48000:
        my_lr = my_lr / 10
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate. 
    optimizer=torch.optim.SGD( model.parameters() , lr=my_lr, momentum = 0.9, weight_decay = 0.0001 )
        
    # set the running quatities to zero at the beginning of the epoch
    running_loss=0
    running_error=0
    num_batches=0
    
    # set the order in which to visit the image from the training set
    shuffled_indices = torch.randperm(45000)
 
    for count in range(0,45000,bs):
    
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        # create a minibatch       
        indices = shuffled_indices[count:count+bs]
        minibatch_data = train_data[indices]
        minibatch_label = train_label[indices]
        
        # send them to the gpu
        minibatch_data=minibatch_data.to(device)
        minibatch_label=minibatch_label.to(device)
        
        # normalize the minibatch (this is the only difference compared to before!)
        inputs = (minibatch_data - mean)/std
        
        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net 
        scores=model( inputs ) 

        # Compute the average of the losses of the data points in the minibatch
        loss =  criterion( scores , minibatch_label) 
        
        # backward pass to compute dL/dU, dL/dV and dL/dW   
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        

        # START COMPUTING STATS
        
        # add the loss of this batch to the running loss
        running_loss += loss.detach().item()
        
        # compute the error made on this batch and add it to the running error       
        error = utils.get_error( scores.detach() , minibatch_label)
        running_error += error.item()
        
        num_batches+=1        
    
    
    # compute stats for the full training set
    total_loss = running_loss/num_batches
    total_error = running_error/num_batches
    elapsed = (time.time()-start)/60
    

    print('epoch=',epoch, '\t time=', elapsed,'min','\t lr=', my_lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    eval_on_dev_set() 
    print(' ')
    
           
    



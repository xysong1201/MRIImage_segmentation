import os
import TrainData_Unet
import torch
import models.UNet as model
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import time
import torch.nn as nn


gpu_id = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

device = torch.device('cuda')

unet = model.UNet(1,256,64)
unet_params = list(unet.parameters())
nb_param=0
for param in unet.parameters():
    nb_param+=np.prod(list(param.data.size()))
print(nb_param)
unet = unet.to(device)

for iteration in range(5):
    start=time.time()
    for sub_idx in range(330):

        T1a_dir = '/home/xiaoyu/MRIdata/T1w/axial/sub{}'.format(sub_idx)
   
        parc5a_dir = '/home/xiaoyu/MRIdata/parc_5/axial/sub{}'.format(sub_idx)
   
        T1a_list = os.listdir(T1a_dir)
  
        parc5a_list = os.listdir(parc5a_dir)

    
        if sub_idx == 0: # set sub0 as test set.
            print('\nT1w Axial slices num:',len(T1a_list))
            print('\nParc5 Axial slices num:',len(parc5a_list))

            continue
        
        print('\nSubject num:',sub_idx)   
        train_data = TrainData_Unet.TrainDataset(T1a_dir=T1a_dir, parc5a_dir = parc5a_dir)
        dataloader = DataLoader(train_data, batch_size = 5, shuffle = True, num_workers = 4)
    
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(unet.parameters() ,lr=0.001)
    
        for epoch in range(0,10):
   
    
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
    print(iteration,'Iteration')
    
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
    
sample=train_data[5]
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



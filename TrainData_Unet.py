from torch.utils.data import Dataset
import os
from skimage import io
import torch
from torchvision import transforms
from PIL import Image





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
        T1a_list = os.listdir(self.T1a_dir)
        parc5a_list = os.listdir(self.parc5a_dir)
        
        T1a_str = T1a_list[idx]
        
        T1a_arr = io.imread(os.path.join(self.T1a_dir, T1a_str))
        T1a_tensor = torch.from_numpy(T1a_arr)
        
        compose_T1 = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize((128,128),interpolation=Image.NEAREST),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        T1a_tensor = torch.unsqueeze(T1a_tensor, dim = 0)
        T1a_tensor = compose_T1(T1a_tensor)
              
        parc5a_str = parc5a_list[idx]
    
        parc5a_arr = io.imread(os.path.join(self.parc5a_dir, parc5a_str))
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


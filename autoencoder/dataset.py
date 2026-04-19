import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir):
        data_names = glob.glob(os.path.join(data_dir, '*f.npy'))
        total_rows = 0
        for i in tqdm(range(len(data_names))):
            features = np.load(data_names[i], mmap_mode='r')  
            total_rows += features.shape[0]

        first_sample = np.load(data_names[0], mmap_mode='r')
        self.data = np.empty((total_rows, first_sample.shape[1]), dtype=first_sample.dtype)
        
        current_idx = 0
        self.data_dic = {}
        for i in tqdm(range(len(data_names))):
            features = np.load(data_names[i])
            name = data_names[i].split('/')[-1].split('.')[0]
            rows = features.shape[0]
            self.data_dic[name] = rows

            self.data[current_idx:current_idx + rows] = features
            current_idx += rows

    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        return data

    def __len__(self):
        return self.data.shape[0] 
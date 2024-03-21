import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Delirium_Dataset(Dataset):
    def __init__(self, data):
        
        self.emr = data['emr']
        self.vitals = data['vitals']
        self.label = data['label']       

    def __len__(self):
        return len(self.label) 


    def __getitem__(self, idx):
        x = {
            'emr': self.emr[idx],
            'vitals': self.vitals[idx],
        }
        y = self.label[idx]

        return x, y

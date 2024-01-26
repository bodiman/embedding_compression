import torch
from torch.utils.data import Dataset

import os
import json
import random

class TextDataset(Dataset):
    def __init__(self, data_path):
        super(TextDataset, self).__init__()

        assert os.path.isfile(data_path), f"Invalid path provided: {data_path}"
        
        with open(data_path, "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'text': self.data[idx]['text']
        }
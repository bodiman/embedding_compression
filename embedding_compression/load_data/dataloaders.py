import torch
from torch.utils.data import Dataset

import os
import json
import random

class DataFormatError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class DistributedTrainingDataset(Dataset):
    def __init__(self, data_path):
        super(DistributedTrainingDataset, self).__init__()

        assert os.path.isdir(data_path), f"Invalid path provided: {data_path}"
        all_files = os.listdir(data_path)
        self.epoch_multiplier = len(all_files)

        for file in all_files:
            if not file.endswith(".json") and not file.startswith("."):
                raise DataFormatError(f"Provided data path must be a directory containing only json files. Found file {file} at {data_path}")

        selected_file = random.choice(all_files)
        self.filepath = f"{data_path}/{selected_file}"
        
        with open(self.filepath, "r") as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "text": self.data[idx]['text']
        }
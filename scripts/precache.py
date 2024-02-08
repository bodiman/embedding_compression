import os
import sys

# Add the parent directory of the scripts directory to the Python path
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..embedding_compression import CompressedEmbedding, TextDataset
from torch.utils.data import DataLoader

import asyncio

import warnings


if __name__ == "__main__":
    cache_data = TextDataset("formatted_data.json")
    cache_dataloader = DataLoader(dataset=cache_data, batch_size=4, shuffle=False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = CompressedEmbedding(0)

    json_array = []

    #loop through datapoints and append vectors as json values. {vector: [1, 2, ...]} kind of thing.
    #If the loop breaks for any reason, combine 

    for i, text in enumerate(cache_dataloader):
        inputs = asyncio.run(model.get_embedding(text))
        print("this ran")
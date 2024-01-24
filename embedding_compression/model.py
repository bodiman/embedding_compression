import os
from dotenv import load_dotenv
load_dotenv()

from .openai_client import client

# def get_embedding(text, model="davinci-001"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# print(len(get_embeddindcg("testing testing 123")))

import torch
import torch.nn as nn
from torch import tanh

"""
Feed Forward Neural Network for compressing ADA 2 embeddings while maintaining the distribution

Attributes
---------

embedding_model: str
    The base openai embedding model to build off of

latent_dims: []int
    number of neurons in the dense layers of the head network
"""
class CompressedEmbedding(nn.Module):
    def __init__(self, embedding_model):
        super(CompressedEmbedding, self).__init__()

        self.client = client
        self.embedding_model = embedding_model
        self.embedding_dim = len(self.get_embedding("Lorem Ipsum"))
        
        self.ff1 = nn.Linear(in_features=1536, out_features=2000, bias=True), 
        self.ff2 = nn.Linear(in_features=2000, out_features=250, bias=True), 
        self.ff3 = nn.Linear(in_features=250, out_features=2000, bias=True), 
        self.ff4 = nn.Linear(in_features=2000, out_features=1536, bias=True)

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return torch.tensor(self.client.embeddings.create(input = [text], model=self.embedding_model).data[0].embedding)
    
    def forward(self, text):
        x = self.get_embedding(text)
        x = tanh(self.ff1(x))
        x = tanh(self.ff2(x))
        x = tanh(self.ff3(x))
        x = tanh(self.ff4(x))
        
        return x
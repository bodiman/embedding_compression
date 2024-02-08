import os
from dotenv import load_dotenv
load_dotenv()

import asyncio

from .openai_client import client

# def get_embedding(text, model="davinci-001"):
#    text = text.replace("\n", " ")
#    return client.embeddings.create(input = [text], model=model).data[0].embedding

# print(len(get_embeddindcg("testing testing 123")))

import torch
import torch.nn as nn
from torch import tanh

from .utils import buffer_text

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
    def __init__(self, latent_dim, embedding_model="text-embedding-ada-002"):
        super(CompressedEmbedding, self).__init__()

        self.client = client
        self.embedding_model = embedding_model

        sample_embedding = asyncio.run(self.client.embeddings.create(input=["Lorem Ipsum"], model=self.embedding_model)).data[0].embedding
        self.embedding_dim = len(sample_embedding)
        self.latent_dim = latent_dim
        
        self.ff1 = nn.Linear(in_features=self.embedding_dim, out_features=self.latent_dim, bias=True) 
        self.ff2 = nn.Linear(in_features=self.latent_dim, out_features=self.embedding_dim, bias=True)

    async def get_embedding(self, text_batch):
        text_batch = text_batch['text']
        text_batch = buffer_text(text_batch, 3*8000) # model supports maximum of 8000ish tokens. Estimating about 3 characters per token to be safe. Use tiktoken for better precision in future

        tasks = [self.client.embeddings.create(input=[text], model=self.embedding_model) for text in text_batch]
        responses = await asyncio.gather(*tasks)

        return torch.tensor([response.data[0].embedding for response in responses])
    
    def forward(self, x):
        x = tanh(self.ff1(x))
        return tanh(self.ff2(x))
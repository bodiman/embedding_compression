from embedding_compression import CompressedEmbedding, TextDataset

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import asyncio

model = CompressedEmbedding(250)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

train_data = TextDataset("./formatted_data.json")
train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for text in train_dataloader:
        """
            1. Input text of size (B,)
            2. Break up inputs and tokenize text to get inputs of size (B + ?, d_0)
            3. Run inputs through model
            4. Calculate gradient from self-prediction
        """

        inputs = asyncio.run(model.get_embedding(text))
        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        reconstruction = model(inputs)
        
        loss = criterion(reconstruction, inputs)

        loss.backward()
        optimizer.step()
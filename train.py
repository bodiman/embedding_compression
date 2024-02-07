from embedding_compression import CompressedEmbedding, TextDataset

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import asyncio

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="embedding-compression",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0005,
    "latent_dimension": 1536,
    "architecture": "autoencoder",
    "dataset": "tiny-wikipedia-json-file",
    "epochs": 2,
    }
)

model = CompressedEmbedding(wandb.config.latent_dimension)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

train_data = TextDataset("./formatted_data.json")
train_dataloader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

for epoch in range(wandb.config.epochs):
    model.train()
    for i, text in enumerate(train_dataloader):
        inputs = asyncio.run(model.get_embedding(text))
        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        reconstruction = model(inputs)
        
        loss = criterion(reconstruction, inputs)

        if i % 1 == 0:
            print(f"[{epoch, i}]: {loss: .{4}f}")
            wandb.log({"MSE": loss})

        loss.backward()
        optimizer.step()

wandb.finish()
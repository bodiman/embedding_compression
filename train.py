from embedding_compression import CompressedEmbedding, TextDataset

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

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
        inputs = model.get_embedding(text)
        inputs = inputs.to(device)
        
        optimizer.zero_grad()

        reconstruction = model(inputs)
        
        loss = criterion(reconstruction, inputs)

        loss.backward()
        optimizer.step()
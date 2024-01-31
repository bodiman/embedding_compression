from embedding_compression import TextDataset

import asyncio


# from torch.utils.data import Dataset, DataLoader

# train_data = TextDataset("./formatted_data.json")
# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

from embedding_compression import CompressedEmbedding

embedding_model = CompressedEmbedding(250)
print(embedding_model.embedding_dim)

x = asyncio.run(embedding_model.get_embedding({"text": ["testing123r454674321rehfglkmdjklnrejlksfd"]}))
print(x)
print(x.shape)
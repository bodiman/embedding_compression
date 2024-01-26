from embedding_compression import TextDataset


# from torch.utils.data import Dataset, DataLoader

# train_data = TextDataset("./formatted_data.json")
# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

from embedding_compression import CompressedEmbedding

embedding_model = CompressedEmbedding(250)
x = embedding_model.get_embedding("test")
print(x)
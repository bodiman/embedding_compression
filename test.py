# from load_data import DistributedTrainingDataset

# from torch.utils.data import Dataset, DataLoader

# train_data = DistributedTrainingDataset("./dataset")
# train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True)

from embedding_compression import CompressedEmbedding

embedding_model = CompressedEmbedding("text-embedding-ada-002")
x = embedding_model.get_embedding("test")
print(x)
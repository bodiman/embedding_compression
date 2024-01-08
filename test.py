from load_data import DistributedTrainingDataset

train_data = DistributedTrainingDataset("./dataset")
print(len(train_data))
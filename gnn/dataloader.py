import random
import numpy as np
import dgl
import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler


class HandGestureDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=32,
        test_split=0.2,
        val_split=0.2,
        shuffle=True,
        random_seed=42
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.shuffle = shuffle
        self.random_seed = random_seed

        # Split dataset into train, test, and validation sets
        dataset_size = len(self.dataset)

        # Generate a random permutation of indices
        np.random.seed(random_seed)
        random_indices = np.random.permutation(dataset_size)

        split_1 = int(np.floor(self.test_split * dataset_size))
        split_2 = int(np.floor(self.val_split * dataset_size))

        self.test_indices, self.val_indices, self.train_indices = random_indices[:split_1], \
            random_indices[split_1:(split_1+split_2)], \
            random_indices[(split_1+split_2):]

        # Define the sample for each set
        self.train_dataset = Subset(dataset, self.train_indices)
        self.test_dataset = Subset(dataset, self.test_indices)
        self.val_dataset = Subset(dataset, self.val_indices)

    @staticmethod
    def collate(batch):
        graphs, labels = zip(*batch)
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.stack(labels).type(torch.float32)  # convert to float32
        return batched_graph, batched_labels

    def get_train_loader(self):
        # Ref: https://docs.dgl.ai/en/1.0.x/generated/dgl.dataloading.GraphDataLoader.html
        return dgl.dataloading.GraphDataLoader(self.train_dataset, batch_size=32, shuffle=self.shuffle, drop_last=False)

    def get_test_loader(self):
        return dgl.dataloading.GraphDataLoader(self.test_dataset, batch_size=32, shuffle=False, drop_last=False)

    def get_val_loader(self):
        return dgl.dataloading.GraphDataLoader(self.val_dataset, batch_size=32, shuffle=False, drop_last=False)

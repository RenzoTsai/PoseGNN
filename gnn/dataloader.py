import random
import numpy as np
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
        indices = list(range(dataset_size))
        split_1 = int(np.floor(self.test_split * dataset_size))
        split_2 = int(np.floor(self.val_split * dataset_size))

        test_indices, val_indices, train_indices = indices[:split_1], indices[split_1:(split_1+split_2)], indices[(split_1+split_2):]

        # Define the sample for each set
        self.train_dataset = Subset(dataset, train_indices)
        self.test_dataset = Subset(dataset, test_indices)
        self.val_dataset = Subset(dataset, val_indices)

    def get_train_loader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def get_test_loader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_val_loader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

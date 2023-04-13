import numpy as np
import torch

from dataloader import *
from dgldataset import HandGestureDataset

if __name__ == '__main__':
    hands_dataset = HandGestureDataset()

    loader = HandGestureDataLoader(hands_dataset, batch_size=32, test_split=0.2, val_split=0.2, shuffle=True)

    train_loader = loader.get_train_loader()
    test_loader = loader.get_test_loader()
    val_loader = loader.get_val_loader()

    sample_data = train_loader.dataset[0]
    print(sample_data)

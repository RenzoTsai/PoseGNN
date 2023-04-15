import numpy as np
import torch

from dataloader import *
from dgldataset import HandGestureDataset

if __name__ == '__main__':
    hands_dataset = HandGestureDataset()

    graph, label = hands_dataset[0]
    print(label)
    # print(graph.ndata['feat'])
    # print(graph.edges())
    # print(type(hands_dataset[0]), type(graph), type(label))

    loader = HandGestureDataLoader(hands_dataset, batch_size=32, test_split=0.2, val_split=0.2, shuffle=True)
    training_set = loader.train_dataset     # Subset of the hands_dataset

    # print(hands_dataset[0], '\n', loader.train_dataset[0])
    # graph_, label_ = loader.train_dataset[0]
    # print(graph_.ndata['feat'], label_)

    train_loader = loader.get_train_loader()
    test_loader = loader.get_test_loader()
    val_loader = loader.get_val_loader()

    for batch_idx, (graph, label) in enumerate(train_loader):
        print(batch_idx, graph, label)


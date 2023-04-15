import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from models import GraphSAGEModel

from dataloader import *
from dgldataset import HandGestureDataset
from utils import accuracy, test

from dgl.data import citegrh

if __name__ == '__main__':

    # Create the dgl dataset
    hand_gesture_dataset = HandGestureDataset()

    # # TODO debug delete this later
    # dataset = SyntheticDataset()
    # graph = hand_gesture_dataset[0][0]
    # print(graph)
    # print(graph.ndata['feat'])
    # print(graph.ndata['feat'].shape)

    # Define data loader
    hand_gesture_dataloader = HandGestureDataLoader(dataset=hand_gesture_dataset)
    train_loader = hand_gesture_dataloader.get_train_loader()
    test_loader = hand_gesture_dataloader.get_test_loader()
    val_loader = hand_gesture_dataloader.get_val_loader()

    # A random GNN model
    num_node_features = hand_gesture_dataset.num_node_features      # Get the number of features for each node - 3
    num_classes = hand_gesture_dataset.num_classes                  # Get the number of classes - 36
    model = GraphSAGEModel(in_feats=num_node_features, n_hidden=32, out_dim=num_classes,
                           n_layers=2, activation=F.relu, dropout=0.5, aggregator_type='gcn')

    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = CrossEntropyLoss()

    # Train the model
    epochs = 10

    # # TODO trial delete later if not needed
    # # Ref: https://docs.dgl.ai/en/1.0.x/generated/dgl.dataloading.GraphDataLoader.html
    # dataloader = dgl.dataloading.GraphDataLoader(hand_gesture_dataset, batch_size=32, shuffle=True, drop_last=False)
    # counter = 0
    # for batched_graph, labels in dataloader:
    #     counter += 1
    #     print(batched_graph)
    #     print(batched_graph.ndata['feat'].shape)
    #     print(counter)

    for epoch in range(epochs):
        # Set model to the train mode
        model.train()
        # Train on batches
        for batch_idx, (bg, labels) in enumerate(train_loader):
            # Forward pass
            features = bg.ndata['feat']
            logits = model(bg, features)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training loss for every 10 iterations
            if batch_idx % 10 == 0:
                print('Epoch {}, Iteration {}, loss {:.4f}'.format(epoch, batch_idx, loss.item()))


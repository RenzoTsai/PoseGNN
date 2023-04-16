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
    epochs = 100

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

            # # Print training loss for every 10 iterations
            # if batch_idx % 10 == 0:
            #     print('Epoch {}, Iteration {}, loss {:.4f}'.format(epoch, batch_idx, loss.item()))

        # Evaluate on validation set
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for batch_idx, (bg, labels) in enumerate(val_loader):
                # Forward pass
                features = bg.ndata['feat']
                logits = model(bg, features)

                # Compute loss
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                # Compute accuracy
                predicted = torch.argmax(logits, dim=1)
                true_labels = torch.argmax(labels, dim=1)
                total_val_acc += (predicted == true_labels).sum().item()

                num_val_batches += 1

            avg_val_loss = total_val_loss / num_val_batches
            avg_val_acc = total_val_acc / len(hand_gesture_dataloader.val_indices)

            print('Epoch {}, Validation Loss {:.4f}, Validation Accuracy {:.4f}'.format(epoch, avg_val_loss,
                                                                                        avg_val_acc))

    # Evaluate on test set
    model.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
    num_test_batches = 0
    with torch.no_grad():
        for batch_idx, (bg, labels) in enumerate(test_loader):
            # Forward pass
            features = bg.ndata['feat']
            logits = model(bg, features)

            # Compute loss
            loss = criterion(logits, labels)
            total_test_loss += loss.item()

            # Compute accuracy
            predicted = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            total_test_acc += (predicted == true_labels).sum().item()

            num_test_batches += 1

        avg_test_loss = total_test_loss / num_test_batches
        avg_test_acc = total_test_acc / len(hand_gesture_dataloader.test_indices)

        print('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(avg_test_loss, avg_test_acc))


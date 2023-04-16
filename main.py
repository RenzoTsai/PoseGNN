from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from gnn.models import GINModel, GAT_Net

from gnn.dataloader import *
from gnn.dgldataset import HandGestureGraphDataset


from gnn.dataloader import HandGestureDataLoader


def train(model, train_loader, val_loader):
    # Define optimizer and loss function
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = CrossEntropyLoss()

    # Train the model
    epochs = 10

    for epoch in range(epochs):
        # Set model to the train mode
        model.train()
        total_train_loss = 0.0
        # Train on batches
        for batch_idx, (bg, labels) in enumerate(train_loader):
            # Forward pass
            features = bg.ndata['feat']
            logits = model(bg, features)

            # Compute loss
            loss = criterion(logits, labels)
            total_train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        total_val_loss = 0.0
        total_val_acc = 0.0
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
                # print(f'The predicted is: {predicted}, the true_label is: {true_labels}')
                # print(f'The correct number was: {(predicted == true_labels).sum()}, the number of all candidates is: {len(true_labels)}, the number of val_loader is: {len(val_loader)}')
                total_val_acc += ((predicted == true_labels).sum()).item()

            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_acc = total_val_acc / len(hand_gesture_dataloader.val_indices)

            # Print training and validation loss for the epoch
            print('Epoch {}, Train Loss {:.4f}, Val Loss {:.4f}, Val Accuracy {:.4f}'.format(epoch,
                                                                                             total_train_loss / len(
                                                                                                 train_loader),
                                                                                             avg_val_loss, avg_val_acc))
    return model


def test(model, test_loader):
    criterion = CrossEntropyLoss()
    # Evaluate on test set
    model.eval()
    total_test_loss = 0.0
    total_test_acc = 0.0
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
            total_test_acc += ((predicted == true_labels).sum()).item()

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_acc / len(hand_gesture_dataloader.test_indices)

        print('Test Loss {:.4f}, Test Accuracy {:.4f}'.format(avg_test_loss, avg_test_acc))


if __name__ == '__main__':

    # Create the dgl dataset
    hand_gesture_dataset = HandGestureGraphDataset()

    # Define data loader
    hand_gesture_dataloader = HandGestureDataLoader(dataset=hand_gesture_dataset)
    train_loader = hand_gesture_dataloader.get_train_loader()
    test_loader = hand_gesture_dataloader.get_test_loader()
    val_loader = hand_gesture_dataloader.get_val_loader()

    num_node_features = hand_gesture_dataset.num_node_feature_dim      # Get the number of features for each node - 3
    num_classes = hand_gesture_dataset.num_classes                  # Get the number of classes - 36

    # Create the model
    # # GraphSAGE model
    # graphsage = GraphSAGEModel(in_feats=num_node_feature_dim, n_hidden=32, out_dim=num_classes,
    #                            n_layers=5, activation=F.relu, dropout=0.5, aggregator_type='gcn')
    # # Train the model
    # graphsage_trained_model = train(graphsage, train_loader, val_loader)
    # # Test the model
    # test(graphsage_trained_model, test_loader)
    #
    # GIN model
    gin = GINModel(in_feats=num_node_features, n_hidden=32, out_dim=num_classes)
    # Train the model
    gin_trained_model = train(gin, train_loader, val_loader)
    # Test the model
    test(gin_trained_model, test_loader)



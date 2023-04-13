import torch


def train(model, train_loader, val_loader, test_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.01,
                                 weight_decay=0.01)
    epochs = 100

    model.train()
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Train on batches
        for data in train_loader:
            optimizer.zero_grad()
            _, out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss / len(train_loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(train_loader)
            loss.backward()
            optimizer.step()

            # Validation
            val_loss, val_acc = test(model, val_loader)

        # Print metrics every 10 epochs
        if epochs % 10 == 0:
            print(f'Epoch {epochs:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc * 100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc * 100:.2f}%')

        test_loss, test_acc = test(model, test_loader)
        print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')

    return model


def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    acc = 0

    for data in loader:
        _, out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y) / len(loader)
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

import torch


def train(model, train_loader, criterion, optimizer):
    running_loss = 0
    # model.train()
    for sub_data in train_loader:  # Iterate over each mini-batch.
        optimizer.zero_grad()  # Clear gradients.
        out = model(sub_data["x"], sub_data["edge_index"])  # Perform a single forward pass.
        loss = criterion(out[sub_data.train_mask],
                         sub_data.y[sub_data.train_mask])  # Compute the loss solely based on the training nodes.
        running_loss += loss.item()  # Compute sum of loss values during an iteration.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    running_loss /= len(train_loader)

    return running_loss


# noinspection PyUnresolvedReferences
def test(model, data):
    # model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with the highest probability.

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask] == data.y[mask]  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / int(mask.sum()))  # Derive ratio of correct predictions.
    return accs


def run(model, train_loader, criterion, test_data, optimizer, epochs=5):
    for epoch in range(1, epochs):
        loss = train(model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)
        train_acc, val_acc, test_acc = test(model, test_data)
        print(f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

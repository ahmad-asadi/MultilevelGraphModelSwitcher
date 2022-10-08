import datetime
import numpy as np

import torch


def train(model, train_loader, criterion, optimizer):
    running_loss = 0
    # model.train()
    for sub_data in train_loader:  # Iterate over each mini-batch.
        optimizer.zero_grad()  # Clear gradients.

        # out = model(sub_data["x"], batch=None)  # Perform a single forward pass, while using KNN_Graph
        out = model(sub_data, batch=None)  # Perform a single forward pass.

        y_pred = -1
        if sub_data["x"][-1][3] > sub_data["x"][-2][3]:
            y_pred = 1

        # print(out.item(), y_pred)
        target = torch.tensor([y_pred])
        target = target.to(torch.float32)

        loss = criterion(out, target)  # Compute the loss solely based on the training nodes.
        running_loss += loss.item()  # Compute sum of loss values during an iteration.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

    running_loss /= len(train_loader)

    return running_loss


def test(model, test_data):
    accs = []
    for sub_data in test_data:
        pred = model(sub_data, batch=None)  # Perform a single forward pass.
        y_pred = -1
        if sub_data["x"][-1][3] > sub_data["x"][-2][3]:
            y_pred = 1

        correct = pred * y_pred > 0  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / len(test_data))  # Derive ratio of correct predictions.
    return np.sum(accs)


def run(model, train_loader, criterion, test_data, optimizer, epochs=5):
    for epoch in range(1, epochs):

        loss = train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer)

        now = datetime.datetime.now()
        if test_data is not None:
            test_acc = test(model=model, test_data=test_data)

            print(now, f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

        else:
            print(now, f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}')

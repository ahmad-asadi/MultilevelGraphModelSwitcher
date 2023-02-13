import datetime
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter


def train(stock_fusion_model, market_fusion_model, train_loader, criterion, optimizer, writer=None, epoch=None):
    running_loss = 0
    # model.train()
    for sub_data in train_loader:  # Iterate over each mini-batch.
        optimizer.zero_grad()  # Clear gradients.

        # out = model(sub_data["x"], batch=None)  # Perform a single forward pass, while using KNN_Graph
        stock_feats = stock_fusion_model(sub_data, batch=None)  # Perform a single forward pass.

        out = market_fusion_model(stock_feats)

        # y_pred = -1
        # if sub_data["x"][-1][3] > sub_data["x"][-2][3]:
        #     y_pred = 1
        y_pred = sub_data["y"]

        # print(out.item(), y_pred.item())
        # target = torch.tensor([y_pred])
        target = y_pred.to(torch.float32)

        loss = criterion(out, target)  # Compute the loss solely based on the training nodes.
        # print(loss.item())

        running_loss += loss.item()  # Compute sum of loss values during an iteration.
        writer.add_scalar("Loss/train", loss, epoch)

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

    running_loss /= len(train_loader)

    return running_loss


def test(stock_fusion_model, market_fusion_model, test_data, writer=None, epoch=None):
    accs = []
    for sub_data in test_data:
        stock_feats = stock_fusion_model(sub_data, batch=None)  # Perform a single forward pass.

        pred = market_fusion_model(stock_feats)

        y_pred = sub_data["y"]

        # print(pred, y_pred)

        correct = torch.argmax(pred) == torch.argmax(y_pred)  # Check against ground-truth labels.
        accs.append(int(correct.sum()) / len(test_data))  # Derive ratio of correct predictions.

    accuracy = np.sum(accs)
    writer.add_scalar("acc/test", accuracy, epoch)

    return accuracy


def run(stock_fusion_model, market_fusion_model, train_loader, criterion, test_data, optimizer, epochs=5):
    writer = SummaryWriter(log_dir="/tmp/")

    for epoch in range(1, epochs):

        loss = train(stock_fusion_model=stock_fusion_model, market_fusion_model=market_fusion_model,
                     train_loader=train_loader, criterion=criterion, optimizer=optimizer, writer=writer,
                     epoch=epoch)

        now = datetime.datetime.now()
        if test_data is not None:
            test_acc = test(stock_fusion_model=stock_fusion_model, market_fusion_model=market_fusion_model,
                            test_data=test_data, writer=writer, epoch=epoch)

            print(now, f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Test Acc: {test_acc:.4f}')

        else:
            print(now, f'Epoch: {epoch:03d}, Train Loss: {loss:.4f}')

        if epoch % 10 == 0:
            writer.flush()

    writer.close()

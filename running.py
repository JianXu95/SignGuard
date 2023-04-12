
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from attacks import *
import tools
import numpy as np
import time


def benignWorker(model, train_loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(train_loader)
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    user_grad = tools.get_gradient_values(model)

    return user_grad, loss.item()

def byzantineWorker(model, train_loader, optimizer, args):
    device = args.device
    attack = args.attack
    model.train()
    criterion = nn.CrossEntropyLoss()
    images, labels = next(train_loader)
    if attack=='label_flip':
        labels = 9 - labels
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    user_grad = tools.get_gradient_values(model)

    return user_grad, loss.item()

# define model testing function
def test_classification(device, model, test_loader):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # pred = output.max(1, keepdim=True)[1]
            correct += (predicted == labels).sum().item()
    acc = 100.0*correct/total
    # print('Test Accuracy: %.2f %%' % (acc))

    return acc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as f
from torch.utils.data import RandomSampler, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, RandomHorizontalFlip, Grayscale, Resize, RandomCrop, ToTensor

import time
START_TIME = time.time()

kernel_size = 3
stride = 1
classes = ('T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle boots')
batch_size = 20
dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor()
)
split = int(len(dataset)*0.8)
train_set, vali_set = random_split(dataset, [split, len(dataset)-split])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
vali_loader = DataLoader(vali_set, batch_size=batch_size, shuffle=True)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        out_10 = 10
        out_5 = 5
        out_16 = 16
        self.conv_10 = nn.Conv2d(in_channels=1, out_channels=out_10, kernel_size=kernel_size, stride=stride)
        self.conv_5 = nn.Conv2d(in_channels=out_10, out_channels=out_5, kernel_size=kernel_size, stride=stride)
        self.conv_16 = nn.Conv2d(in_channels=out_5, out_channels=out_16, kernel_size=kernel_size, stride=stride)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 40)
        self.fc3 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv_10(x)
        x = f.relu(x)
        x = self.pool(x)

        x = self.conv_5(x)
        x = f.relu(x)

        x = self.conv_16(x)
        x = f.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = f.relu(x)

        x = self.fc2(x)
        x = f.relu(x)

        x = self.fc3(x)
        return x

if __name__ == "__main__":
    l_rate = 0.01  # 0.01
    epochs = 15
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), l_rate)
    fig, ax = plt.subplots()

    train_loss = []
    vali_loss = []
    print("Training...")
    for epoch in range(epochs):
        train_loss_ep = 0
        vali_loss_ep = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_ep += loss.item()
        train_loss.append(train_loss_ep)

        for i, data in enumerate(vali_loader):
            inputs, labels = data
            outputs = net(inputs)
            vali_loss_ep += criterion(outputs, labels).item()

        vali_loss.append(vali_loss_ep)

    ax.plot(range(epochs), train_loss, label='Training loss')
    ax.plot(range(epochs), vali_loss, label='Validation loss')
    ax.legend()
    ax.set_title("Loss vs Epochs")

    PATH = './mnist_net.pth'
    torch.save(net.state_dict(), PATH)
    # net = CNN()
    # net.load_state_dict(torch.load(PATH, weights_only=True))

    total = 0
    correct = 0
    loss = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.append(criterion(outputs, labels))

    print("Test set accuracy:", correct/total)
    # elapsed = time.time() - START_TIME
    # minutes = elapsed // 60
    # seconds = int(elapsed) % 60
    # print(f"--- Executed in {minutes} minutes {seconds} seconds ---")
    plt.show()

    print("done")

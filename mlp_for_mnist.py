import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import cv2
import numpy as np
from utils import UnNormalize

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.classifier(x)
        return x


def run(device):
    # prepare training and testing data
    training_dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    testing_dataset = torchvision.datasets.MNIST('./data', train=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=3)

    # create checkpoint dir
    save_checkpoint = True
    save_dir = "checkpoint/mnist"
    os.makedirs(save_dir, exist_ok=True)

    # model and optimizer
    model = Net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())

    # training
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = F.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()), end='')

        if save_checkpoint:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(save_dir, "epoch%d.pth"%epoch))

    print()

    # testing
    print("Testing...")
    correct = 0
    model.eval()
    un = UnNormalize((0.1307,), (0.3081,))

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)

        pred = logits.argmax(dim=1, keepdim=True)           # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        # visualization
        pred = pred.view(-1).cpu().numpy()
        gt = target.cpu().numpy()
        print("The prediction result: ", end="")
        print(pred, end="")
        print(". Groundtruth:", end="")
        print(gt)

        original_image = data
        original_image = original_image.cpu().numpy()

        for i in range(original_image.shape[0]):
            image = original_image[i]
            grey_layer = image[0]
            image = np.stack([grey_layer, grey_layer, grey_layer], axis=2)
            cv2.imshow("%d"%i, image)
        cv2.waitKey()


    accuracy = 100. * correct / len(test_loader.dataset)
    print('Accuracy: {}/{} ({:.5f}%)'.format(correct, len(test_loader.dataset), accuracy))
    print("Done!")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(device)

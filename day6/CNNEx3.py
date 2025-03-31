import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

cifar10_train = datasets.CIFAR10('CIFAR10_data/',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) #(R,G,B)
                                 ]))

cifar10_test = datasets.CIFAR10('CIFAR10_data/',
                                 train=False,
                                 download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # (R,G,B)
                                ]))

trainloader = DataLoader(cifar10_train, batch_size=16, shuffle=True)
testloader = DataLoader(cifar10_test, batch_size=16, shuffle=False)

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        self.pool = nn.MaxPool2d(2,2, padding=1)

        self.fc1 = nn.Linear(128*2*2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 84)
        self.fc4 = nn.Linear(84, 10)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        x = F.relu(self.conv1(x)) #(32,28,28)
        x = self.pool(x) #(32,15,15) 패딩을 1로 줘서

        x = F.relu(self.conv2(x)) #(64,11,11)
        x = self.pool(x) #(64,6,6)

        x = F.relu(self.conv3(x)) #(128,2,2)
        x = self.pool(x) #(128,2,2)

        x = x.view(-1, 128*2*2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = self.fc4(x)
        return y

model =CNNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(20):
#     running_loss = 0.
#     for idx, (x_train, y_train) in enumerate(trainloader):
#         optimizer.zero_grad()
#         hypothesis = model(x_train)
#         loss = loss_func(hypothesis, y_train)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         if idx % 2000 == 0:
#             print(f'epoch:{epoch+1} / {idx+1} loss:{running_loss/2000:.5f}')
#             running_loss = 0.

PATH = './cifar_net.pt'
# torch.save(model.state_dict(), PATH)

model.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for (x_test, y_test) in testloader:
        outputs = model(x_test)
        prediction = torch.argmax(outputs.data, dim=1)
        total += y_test.size(0)
        correct += (prediction == y_test).sum().item()

print(f'accuracy:{correct / total * 100:2.2f}%')
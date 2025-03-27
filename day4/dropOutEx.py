import numpy as np
from sympy import evaluate
from torchvision import datasets
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


train_set = datasets.FashionMNIST(root='FashionMNIST_data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

test_set = datasets.FashionMNIST(root='FashionMNIST_data/',
                           train=False,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

from torch.utils.data import DataLoader

batch_size=100
train_loader=DataLoader(train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

test_loader=DataLoader(test_set,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
class ImageNN(nn.Module):
    def __init__(self, drop_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

        self.dropout_p = drop_p

    def forward(self, x):
        x = x.view(-1, 784)
        out = F.relu(self.fc1(x))
        out = F.dropout(out, p=self.dropout_p, training=self.training)# dropout 과정은 학습할 때만 필요, 실제로 쓸 땐 사용하면 안됨 (dropout의 매개변수 training=True)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=self.dropout_p, training=self.training)
        y = self.fc3(out)
        return y

import torch.optim as optim
model = ImageNN(drop_p=0.2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train() # train모드 (48번라인)
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
    model.eval() # 48번라인의 train모드가 False가 됨 (dropout은 실제 사용할 땐 꺼야하므로 eval()로 사용)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            hypothesis = model(x_test)
            test_loss += F.cross_entropy(hypothesis, y_test).item()
            pred = torch.argmax(hypothesis, dim=1)
            correct += pred.eq(y_test.view_as(pred)).sum().item() # pred.eq(y_test.view_as(pred)) - 같은지 확인하는(pred == y_test 임. 즉, bool값을 리턴)

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, test_accuracy

for epoch in range(20):
    train(model, train_loader, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print(f'epoch:{epoch+1}, loss:{test_loss:.4f}, accuracy:{test_accuracy:2.2f}%')


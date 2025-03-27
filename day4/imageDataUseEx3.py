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

# dataiter = iter(train_loader)
# next(dataiter)
# images, labels = next(dataiter)
# print(images.shape) # torch.Size([16, 1, 28, 28])
# print(labels.shape)  # torch.Size([16])

# print(train_set.classes)  # FashionMNIST의 클래스 목록 확인 - ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(len(train_set.classes))  # 총 클래스 개수 확인 - 10


# from torchvision import utils
#
# img = utils.make_grid(images)
# npimg = img.numpy()
# print(npimg.shape) # (3, 62, 242) 이미지를 통합시켜놓은 것. 3이 채널, 62* 242 짜리 이미지. 보통 채널이 뒤에 위치하기 때문에 옮겨야함
# print(np.transpose(npimg, (1,2,0)).shape) # (62, 242, 3)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10,7))
# plt.imshow(np.transpose(npimg, (1,2,0)))
# print(labels)
# plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
class ImageNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)# 원본을 받아서 들어온 데이터를 여기서 직렬화시킴
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        y = self.fc3(out)
        return y

import torch.optim as optim
model = ImageNN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, train_loader, optimizer):
    model.train()
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader):
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


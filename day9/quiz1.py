# FashionMNIST
# 오토인코더
# linear
# imshow에서 첫번째 열은 원본이미지 5개, 두번째 열은 모델예측이미지 5개

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


train_epochs = 10
batch_size = 64 # 가로 크기?
learning_rate = 0.005

mnist_train = dset.FashionMNIST(root='FashionMNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.FashionMNIST(root='FashionMNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self,x):
        eoutput = self.encoder(x)
        doutput = self.decoder(eoutput)
        return eoutput, doutput # 이 코드에서 eoutput을 쓰는 건 아니지만 다른 코드에서는 활용 가능도 중요


autoencoder = AutoEncoder()
loss_func = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

def train(autoencoder, train_loader):
    for x_data, _ in train_loader:
        x_data = x_data.view(-1, 784)

        eoutput, doutput = autoencoder(x_data) # 마찬가지로 여기서는 doutput만 사용
        loss = loss_func(doutput, x_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

view_data = mnist_train.data[:5].view(-1, 28*28)
view_data = view_data.type(torch.FloatTensor) / 255.

import numpy as np

for epoch in range(train_epochs):
    train(autoencoder, data_loader)

    _, doutput = autoencoder(view_data)

    print(f'epoch:{epoch+1}')
    fig, axes = plt.subplots(2,5,figsize=(5,2))
    for i in range(5):
        img = np.reshape(view_data.data.numpy()[i], (28, 28))
        axes[0][i].imshow(img, cmap='gray')
        axes[0][i].set_xticks(())
        axes[0][i].set_yticks(())

    for i in range(5):
        img = np.reshape(doutput.data.numpy()[i], (28, 28))
        axes[1][i].imshow(img, cmap='gray')
        axes[1][i].set_xticks(())
        axes[1][i].set_yticks(())
plt.show()
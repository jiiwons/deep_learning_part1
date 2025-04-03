import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

train_epochs = 20
batch_size = 100
learning_rate = 0.0002

mnist_train = dset.MNIST(root='MNIST_data/',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
mnist_test = dset.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True)

class AutoEncoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
        )
        self.decoder = nn.Sequential(
            nn.Linear(20,256),
            nn.ReLU(),
            nn.Linear(256, 784)
        )

    def forward(self,x):
        x = x.view(batch_size, -1)
        eoutput = self.encoder(x)
        y = self.decoder(eoutput).view(batch_size, 1, 28, 28) # 디코더로 나온 걸 이미지로 변환
        return y

AEModel = AutoEncoderNet()
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(AEModel.parameters(), lr=learning_rate)

for epoch in range(train_epochs):
    for idx, (x_data, y_data) in enumerate(data_loader):
        optimizer.zero_grad()
        hypothesis = AEModel(x_data)
        loss = loss_func(hypothesis, x_data) # y_data가 아니라 x_data가 들억나느 이유는??
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(f'loss:{loss.item():.4f}')

out_img = torch.squeeze(hypothesis.data)
for i in range(3):
    plt.subplot(121)
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.subplot(122)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


train_epochs = 3
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

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # out_channels=16 → 32 → 64 → 128 → 256 (점점 깊은 특징 추출)
        # padding=1 (출력 크기를 유지)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), # in_channels=1 (MNIST는 흑백 이미지)
            nn.ReLU(),
            nn.BatchNorm2d(16), # ResNet에서 사용한 개념, 학습안정화, out_channels의 크기(CNN의 출력 채널 수)와 맞춰줘야함
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), # 64 * 14 * 14
                                                # 2×2 영역에서 최대값을 선택하여 크기 축소

        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), # 128 * 7 * 7
            nn.Conv2d(128, 256, 3, padding=1), # 256 * 7 * 7
            nn.ReLU(),
        )

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        y = out.view(batch_size, -1)
        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            # 업샘플링(UpSampling) 을 수행하는 전치 합성곱(Deconvolution)
            # stride=2 → 크기를 2배로 확대
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), # 128 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 3, 1, 1), # 16 * 14 * 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), # 1 * 28 * 28
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(batch_size, 256, 7, 7)
        out = self.layer1(x)
        y = self.layer2(out)
        return y

encoder = Encoder()
decoder = Decoder()
loss_func = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(parameters, lr=learning_rate)

for epoch in range(train_epochs):
    for idx, (x_data, _) in enumerate(data_loader):
        optimizer.zero_grad()

        eoutput = encoder(x_data)
        hypothesis = decoder(eoutput)

        loss = loss_func(hypothesis, x_data)
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            print(f'{epoch+1} / {idx+1} loss:{loss.item():.4f}')

out_img = torch.squeeze(hypothesis.data)
for i in range(3):
    plt.subplot(121)
    plt.imshow(torch.squeeze(x_data[i]).numpy(), cmap='gray')
    plt.subplot(122)
    plt.imshow(out_img[i].numpy(), cmap='gray')
    plt.show()
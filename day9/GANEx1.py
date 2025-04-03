import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

epochs = 30
batch_size = 100

trainset = dset.FashionMNIST(root='FashionMNIST_data/',
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(), # 데이터를 [0,1] 범위로 변환
                                 transforms.Normalize((0.5,), (0.5,)) # -1~1 범위로 정규화(Tanh 활성화 함수와 맞추기 위함)
                             ]))
train_loader = DataLoader(dataset=trainset,
                          batch_size=batch_size,
                          shuffle=True)

# Generator는 랜덤한 노이즈(64차원 벡터)를 입력으로 받아 28*28(784차원)크기의 가짜 이미지를 생성
Generator = nn.Sequential(
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 784),
    nn.Tanh() # 출력 범위를 -1~1 범위로 맞춤
)

# Discriminator는 입력 이미지를 받아 진짜(1)인지 가짜(0)인지 판별
Discriminator = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2), # 죽은 뉴런 방지
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256,1), # 출력층 (1개의 확률값 출력:1이면 진짜, 0이면 가짜)
    nn.Sigmoid() # 출력을 0~사이의 확률로 변환
)

loss_func = nn.BCELoss() # Binary Cross Entropy Loss, 판별자의 실제 정답과 출력값의 차이를 계산
d_optimizer = optim.Adam(Discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(Generator.parameters(), lr=0.0002)

for epoch in range(epochs):
    for (image, _) in train_loader:
        image = image.view(batch_size, -1) # 28×28 이미지를 784차원 벡터로 변환([100, 1, 28, 28] → [100, 784])
        real_label = torch.ones(batch_size, 1) # 진짜 이미지의 정답(1)
        fake_label = torch.zeros(batch_size, 1) # 가짜 이미지의 정답(0)

        ## Discriminator 학습
        outputs = Discriminator(image) # FashionMNIST 이미지를 판별
        d_loss_real = loss_func(outputs, real_label) # 진짜 이미지의 손실 계산

        z = torch.randn(batch_size, 64) # 랜덤 노이즈 생성(Generator 입력값)
        fake_images = Generator(z) # 가짜 이미지 생성 
        outputs = Discriminator(fake_images) # 가짜 이미지 판별
        d_loss_fake = loss_func(outputs, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward() # 역전파 수행 후 Discriminator 업데이트
        d_optimizer.step()

        ## Generator학습
        fake_images = Generator(z) # 가짜 이미지 생성
        outputs = Discriminator(fake_images) # Discriminator가 가짜 이미지를 판별한 값 
        g_loss = loss_func(outputs, real_label) # 가짜 이미지를 진짜라고 속이는 방향으로 학습
        g_optimizer.zero_grad()
        g_loss.backward()

    print(f'epoch:{epoch+1}, d_loss:{d_loss.item():.4f}, g_loss:{g_loss.item():.4f}')

# 랜덤 노이즈를 입력으로 받아 새로운 이미지를 생성하고, 3개의 이미지를 출력
z = torch.randn(batch_size, 64)
fake_images = Generator(z)

import numpy as np
for i in range(3):
    plt.subplot(131+i)
    fake_image_img = np.reshape(fake_images.data.numpy()[i], (28,28)) # 784차원 벡터를 28x28로 변환
    plt.imshow(fake_image_img, cmap='gray')

plt.show()


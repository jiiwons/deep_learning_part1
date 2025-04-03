# Denoising AutoEncoder(DAE) : 노이즈가 있는 이미지에서 원복 이미지를 복원하는 모델
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim


train_epochs = 20
# batch_size = 64
# learning_rate = 0.005
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

# 노이즈 추가 함수
def add_noise(img):
    noise = torch.randn(img.size()) * 0.3 # 노이즈 강도 조절(0.3 정도의 가중치 부여)
    noise_img = img + noise # 원본 이미지에 노이즈 추가
    return noise_img

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 인코더 : 차원 축소(입력 이미지를 점점 더 작은 크기의 벡터로 압축)
        self.encoder = nn.Sequential(
            nn.Linear(784, 128), # 28x28 이미지를 128차원으로 압축
            nn.ReLU(),
            nn.Linear(128, 64), # 64차원으로 압축 
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3), # 가장 압축된 3차원 표현
        )
        # 디코더 : 노이즈 제거(압축된 벡터를 다시 원래 이미지 크기로 복원)
        self.decoder = nn.Sequential(
            nn.Linear(3, 12), # 압축된 표현에서 다시 확장
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784), # 원본 이미지 크기(28*28)로 복원
            nn.Sigmoid() # 픽셀 값을 0~1 범위로 변환
        )

    def forward(self,x):
        eoutput = self.encoder(x)
        doutput = self.decoder(eoutput)
        return eoutput, doutput


autoencoder = AutoEncoder()
loss_func = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

def train(autoencoder, train_loader):
    avg_loss = 0

    for (image, _) in train_loader:
        x_data = add_noise(image) # 노이즈가 추가된 이미지를 생성 
        x_data = x_data.view(-1, 784) # 이미지를 1D 벡터로 변환 
        y = image.view(-1, 784) 

        _, decoded = autoencoder(x_data) # Autoencoder를 통과하여 복원된 이미지를 얻음
        loss = loss_func(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    return avg_loss / len(train_loader)

for epoch in range(train_epochs):
    loss = train(autoencoder, data_loader)
    print(f'epoch:{epoch+1} loss:{loss:.4f}')

# 샘플 데이터 변환 및 복원 
sample_data = mnist_train.data[1].view(-1, 784).type(torch.FloatTensor)/255.

original_x = sample_data[0] # 샘플 데이터에서 하나의 이미지 가져옴 
noise_x = add_noise(original_x)  # 노이즈 추가 
_, recovered_x = autoencoder(noise_x) # autoencoder를 통해 복원

fig, axes = plt.subplots(1, 3, figsize=(10,10))
import numpy as np
original_img = np.reshape(original_x.data.numpy(), (28,28))
noise_img = np.reshape(noise_x.data.numpy(), (28,28))
recovered_img = np.reshape(recovered_x.data.numpy(), (28,28))

axes[0].set_title('original')
axes[0].imshow(original_img, cmap='gray')
axes[1].set_title('noise and image')
axes[1].imshow(noise_img, cmap='gray')
axes[2].set_title('recovered image')
axes[2].imshow(recovered_img, cmap='gray')
plt.show()
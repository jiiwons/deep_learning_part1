import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


train_epochs = 10
batch_size = 100

mnist_train = dset.MNIST(root='MNIST_data',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_test = dset.MNIST(root='MNIST_data',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # 입력크기:(batch, 1, 28, 28) 출력크기:(batch, 32, 28, 28) 패딩을 1을 줬기 때문에 크기가 유지됨
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 2×2 Max Pooling 적용 → 크기 절반 감소
            # 출력 크기 : (batch, 32, 14, 14) (위에서 batch_size=100으로 정의했으므로 batch는 100)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 입력 채널:32 -> 이전 레이어의 출력 채널/ 입력 크기:(batch, 32, 14, 14) 출력 크기:(batch, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 크기 절반 감소
            # 출력 크기 : (batch, 64, 7, 7)
        )

        self.fc = nn.Linear(64 * 7 * 7, 10) # CNN에서 나온 64개의 7*7특징을 1D벡터로 변환, 10개의 출력 ->MNIST 숫자 예측

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # -1은 자동으로 남은 차원을 펼쳐서 일렬로 만들라는 의미(배치 크기(batch size)를 제외한 나머지 차원들을 하나의 차원으로 펼친다(flatten))
        y = self.fc(out)
        return y

model = CNNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
total_batch = len(data_loader)

for epoch in range(train_epochs):
    avg_loss = 0
    for x_train, y_train in data_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

        avg_loss += loss/total_batch
    print(f'epoch:{epoch+1} avg_loss:{avg_loss:.4f}')

    with torch.no_grad():
        x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float()
        y_test = mnist_test.test_labels
        prediction = model(x_test)
        correction_prediction = torch.argmax(prediction, dim=1)==y_test
        accuracy = correction_prediction.float().mean()
        print('accuracy:{:2.2f}'.format(accuracy.item()*100))
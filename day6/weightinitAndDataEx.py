import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.init as init

batch_size = 100
learning_rate = 0.001
total_epochs = 10

fmnist_train = dset.FashionMNIST('FashionMNIST_data/',
                                 train=True,
                                 transform=transforms.Compose([
                                    transforms.Resize(34),
                                     transforms.CenterCrop(28),
                                     transforms.Lambda(lambda x:x.rotate(90)),
                                     transforms.ToTensor()
                                 ]),
                                 target_transform=None,
                                 download=True)

fmnist_test = dset.FashionMNIST('FashionMNIST_data/',
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 target_transform=None,
                                 download=True)

train_loader = DataLoader(fmnist_train,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(fmnist_test,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

class CNNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Clayer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fclayer = nn.Sequential(
            nn.Linear(64*7*7, 100),
            nn.ReLU(),
            nn.Linear(100,10),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)


    def forward(self,x):
        out = self.Clayer(x)
        out = out.view(batch_size, -1)
        y = self.fclayer(out)
        return y

model = CNNet2()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay = 0.000001) # 실행할 때마다 weight_decay만큼 깎아먹어서 오버피팅을 줄이도록

for i in range(total_epochs):
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()

    print(f'epoch:{i+1}, loss:{loss.item():.4f}')
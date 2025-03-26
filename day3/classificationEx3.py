import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_wine
import numpy as np


wine = load_wine()
print(wine)

import pandas as pd

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df.info()

wine_data = wine.data[0:130]
wine_target = wine.target[0:130]
print(wine_target)
print(np.unique(wine_target))

# train, test 를 8:2로 batch_size=16
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, test_size=0.2, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).long()

dataset = TensorDataset(x_train, y_train) # TensorDataset() → 입력(x_train)과 정답 레이블(y_train)을 묶어주는 역할
train_loader = DataLoader(dataset, batch_size=16, shuffle=True) # DataLoader() → 미니배치 단위로 데이터를 제공하는 역할

class CNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 96)
        self.fc2 = nn.Linear(96, 96)
        self.fc3 = nn.Linear(96, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = F.relu(self.fc5(out))
        y = self.fc6(out)
        return y

model = CNet()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    total_loss = 0

    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f'epoch:{epoch+1}, total_loss:{total_loss:.4f}')

print()
prediction = torch.max(model(x_test), dim=1)[1]
accuracy = (prediction == y_test).float().mean()
print('accuracy:{:.4f}'.format(accuracy.item()))
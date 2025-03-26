import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
#
# # 마지막 컬럼 0:비당뇨, 1:당뇨
# df = pd.read_csv('diabetes.csv')
# df.info()
#
# data = torch.from_numpy(df.values).float()
# x_train = data[:, :-1]
# y_train = data[:, -1:]
#
# print(x_train.shape, y_train.shape)
#
# # hidden layer = 2, batch_size =32, dataset상속
#
# dataset = TensorDataset(x_train, y_train)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#
# class LogisticModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.linear = nn.Linear(input_size, input_size)
#         self.sigmoid = nn.Sigmoid()
#         self.linear = nn.Linear(input_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         y = self.sigmoid(self.linear(x))
#         return y
#
# model = LogisticModel(input_size=x_train.size(-1), output_size=y_train.size(-1))
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
# loss_func = nn.BCELoss()
#
# for epoch in range(2000):
#     for batch_idx, (x_train, y_train) in enumerate(dataloader):
#         hypothesis = model(x_train)
#         loss = loss_func(hypothesis, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if epoch % 100 == 0:
#             print('epoc:{} cost:{:.4f}'.format(epoch + 1, loss.item()))
#
#
# with torch.no_grad():
#     hypothesis = model(x_train)
#     prediction = (hypothesis > 0.5).float()
#     accuracy = (prediction ==y_train).float().mean()
#     print('hypothesis:\n{}\nprediction:\n{}\ntarget:\n{}\naccuracy:{:.4f}'.format(
#         hypothesis.numpy(), prediction.numpy(), y_train.numpy(), accuracy.item()
#     ))
# accuracy:0.7742


## 강사님 코드
import numpy as np
from torch.utils.data import Dataset


class DiabetsDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 수치로만 가져올 땐 numpy가 효율적
        xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        # print(xy.shape)
        self.len_size = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __len__(self):
        return self.len_size

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

dataset = DiabetsDataset()
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class LogisticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 50)
        self.l2 = nn.Linear(50, 10)
        self.l3 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y = self.sigmoid(self.l3(out2))

        return y

import torch.optim as optim

model = LogisticModel()
loss_func = nn.BCELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001) # adam으로 하면 조금 더 성능 개선
for epoch in range(5000):
    for idx, data in enumerate(train_loader):
        x_train, y_train = data
        hypothesis = model(x_train)
        loss = loss_func(hypothesis, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        prediction = hypothesis > torch.FloatTensor([0.5])
        correction_prediction = prediction.float() == y_train
        accuracy = correction_prediction.sum().item() / len(correction_prediction)
        print('epoch:{} loss:{:.4f} accuracy:{:2.2f}'.format(
            epoch, loss.item(), accuracy * 100
        ))
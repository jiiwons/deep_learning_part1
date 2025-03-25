# linear regression 모델 구성
# a. batch size는 3으로 한다.
# b. model은 nn.Module을 상속받아 구성한다.
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
x_train = torch.FloatTensor([[73,80,75,65],
                            [93,88,93,88],
                            [89,91,90,76],
                            [96,98,100,99],
                            [73,65,70,100],
                             [84,98,90,100]])

y_train = torch.FloatTensor([[152],[185],[180],[196],[142],[188]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

class MLRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,1)

    def forward(self,x):
        return self.linear(x)

model = MLRegressionModel()
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

import torch.nn.functional as F

for epoch in range(2000):
    for batch_idx,(x_train, y_train)in enumerate(dataloader):
        hypothesis = model(x_train)
        loss = F.mse_loss(hypothesis, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch%100==0:
            print('epoc:{} cost:{:.4f}'.format(epoch+1, loss.item()))
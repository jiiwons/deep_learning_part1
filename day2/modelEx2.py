import torch

x_train = torch.FloatTensor([[73,80,75],
                            [93,88,92],
                            [89,91,90],
                            [96,98,100],
                            [73,65,70]])

y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True) # 데이터로더 객체는 이터레이터를 포함하고 있음. 2개씩 잘라서 전달하게 됨
print(dataloader)
print()

for data in dataloader:
    print(data, end='\n\n')
print()

import torch.nn as nn
import torch.optim as optim

model = nn.Linear(3,1)
optimizer = optim.SGD(model.parameters(), lr = 1e-5)

import torch.nn.functional as F

for epoch in range(20):
    for batch_idx, data in enumerate(dataloader):
        batch_x, batch_y = data
        hypothesis = model(batch_x)
        loss = F.mse_loss(hypothesis, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'{epoch+1}/{batch_idx+1}, loss:{loss.item():.4f}')
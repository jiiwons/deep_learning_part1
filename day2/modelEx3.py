import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader,Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73,80,75],
                    [93,88,92],
                    [89,91,90],
                    [96,98,100],
                    [73,65,70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for data in dataloader:
    print(data, end='\n\n')

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
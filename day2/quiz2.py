import pandas as pd
from sklearn.datasets import load_breast_cancer

from day2.logisticEx2 import optimizer, hypothesis

cancer = load_breast_cancer()
print(cancer.keys())
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['class'] = cancer.target
df.info()

cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness','mean concavity',
        'worst radius','worst perimeter','worst concavity','worst texture', 'class']

import torch
import torch.nn as nn
import torch.optim as optim

data = torch.from_numpy(df[cols].values).float()
x_train = data[:, :-1]
y_train = data[:, -1:]

print(x_train.shape, y_train.shape)

class LogisticModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.linear(x))
        return y

model = LogisticModel(input_size=x_train.size(-1), output_size=y_train.size(-1))
loss_func = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00005)


for epoch in range(200000):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10000 == 0:
        print(f'epoch:{epoch+1} loss:{loss.item():.4f}')

correction_cnt = (y_train ==(hypothesis>0.5)).sum()
print('accuracy : {:2.1f}%'.format(correction_cnt/float(y_train.size(0))*100))
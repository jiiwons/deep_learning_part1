import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

import torch.nn as nn
class LogisticClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

import torch.nn.functional as F
model = LogisticClass()
optimizer = optim.SGD(model.parameters(), lr=1)

for epoch in range(1000):
    hypothesis = model(x_train)
    loss = F.binary_cross_entropy(hypothesis, y_train)

    # 경사하강법 적용
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 ==0:
        prediction = hypothesis > torch.FloatTensor([0.5])
        correction_prediction = prediction.float() == y_train # 정답 여부 확인
        accuracy = correction_prediction.sum().item() / len(correction_prediction)
        print(f'epoch{epoch+1} loss : {loss.item():.4f} accuracy : {accuracy*100:.2f}')

hypothesis = model(x_train)
print(hypothesis)
prediction = hypothesis > torch.FloatTensor([0.5])
print(prediction)
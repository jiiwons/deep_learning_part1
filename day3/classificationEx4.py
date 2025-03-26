import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


iris = pd.read_csv('iris.csv')
iris.info()
print(iris.head(10))
print(iris.Name)
print(iris.Name.unique())

mappings = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}
iris['Name'] = iris['Name'].map(mappings)
print(iris.head())

x = iris.drop('Name', axis=1).values
y = iris['Name'].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16,12)
        self.fc3 = nn.Linear(12, 3)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        y = self.fc3(out)
        return y

model = NN()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    hypothesis = model(x_train)
    loss = loss_func(hypothesis, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(epoch, loss.item())

preds = []
for val in x_test:
    hypothesis = model(val)
    preds.append(hypothesis.argmax().numpy())

df = pd.DataFrame({'target':y_test, 'pred':preds})
df['correct']=[1 if corr==pred else 0 for corr, pred in zip(df['target'], df['pred'])]
print(df)
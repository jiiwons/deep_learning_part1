import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

root = 'images/'

dataset = ImageFolder(root=root,
                      transform=transforms.Compose([
                          transforms.Resize([224, 224]),
                          transforms.ToTensor()
                      ]))

data_loader = DataLoader(dataset,
                         batch_size=32,
                         shuffle=True)

print(dataset.classes)

images, labels = next(iter(data_loader))
print(images.shape)
print(labels.shape)
print(labels)

labels_map = {v:k for k, v in  dataset.class_to_idx.items()}
print(labels_map)
print(len(dataset))

from torch.utils.data import random_split

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_data, test_data = random_split(dataset, [train_size, test_size])
batch_size = 32

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(test_data,
                          batch_size=batch_size,
                          shuffle=False)

from torchvision import models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

model = models.vgg16(pretrained=True)
#print(model)

for param in model.parameters():
    #print(param.requires_grad)
    param.requires_grad = False

import torch.nn as nn

fc = nn.Sequential(
    nn.Linear(7 * 7 * 512, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

model.classifier = fc
print(model)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()

def model_train(model, data_loader, loss_fn, optimizer):
    model.train()
    running_loss = 0
    corr = 0

    for idx, (x_data, y_data) in enumerate(data_loader):
        optimizer.zero_grad()
        hypothesis = model(x_data)
        loss = loss_fn(hypothesis, y_data)
        loss.backward()
        optimizer.step()
        pred = hypothesis.argmax(dim=1)
        corr += pred.eq(y_data).sum().item()

        running_loss += loss.item() * x_data.size(0)
        print(f'{idx}/{len(data_loader)}')

    acc = corr / len(data_loader.dataset)
    return running_loss / len(data_loader.dataset), acc

def model_evaluate(model, data_loader, loss_fn):
    model.eval()
    with torch.no_grad():
        corr = 0
        running_loss = 0
        for x_test, y_test in data_loader:
            hypothesis = model(x_test)
            pred = hypothesis.argmax(dim=1)
            corr += torch.sum(pred.eq(y_test)).item()
            running_loss += loss_fn(hypothesis, y_test).item() * x_test.size(0)

        acc = corr / len(data_loader.dataset)
        return running_loss / len(data_loader.dataset), acc

num_epochs = 10
model_name = 'vgg16-trained'
min_loss = np.inf

for epoch in range(num_epochs):
    train_loss, train_acc = model_train(model, train_loader, loss_func, optimizer)
    val_loss, val_acc = model_evaluate(model, test_loader, loss_func)

    if val_loss < min_loss:
        print(f'val_loss has been improved from {min_loss:.5f} to {val_loss:.5f}. saving model!!!!!!')
        min_loss = val_loss
        torch.save(model.state_dict(), f'{model_name}.pth')
    print(f'epoch:{epoch+1} loss:{train_loss:.5f}, acc:{train_acc:.5f}'
          f' val_loss:{val_loss:.5f}, val_acc:{val_acc:.5f}')

model.load_state_dict(torch.load(f'{model_name}.pth'))
final_loss, final_acc = model_evaluate(model, test_loader, loss_func)
print(f'evaluation loss:{final_loss:.5f} evaluation accuracy:{final_acc:.5f}')

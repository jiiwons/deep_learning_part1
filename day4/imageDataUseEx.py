import numpy as np
from torchvision import datasets
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_set = datasets.MNIST(root='MNIST_data/',
                           train=True,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

test_set = datasets.MNIST(root='MNIST_data/',
                           train=False,
                           download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))

from torch.utils.data import DataLoader


batch_size=16
train_loader=DataLoader(train_set, batch_size=batch_size)
test_loader=DataLoader(test_set, batch_size=batch_size)

dataiter = iter(train_loader)
next(dataiter)
images, labels = next(dataiter)
print(images.shape) # torch.Size([16, 1, 28, 28]) 16은 데이터 16개씩 넘겨줌을 의미(batch_size), 1은 channel, 28은 height,width
print(labels.shape) # torch.Size([16]) 1차원 형태

from torchvision import utils

img = utils.make_grid(images)
npimg = img.numpy()
print(npimg.shape) # (3, 62, 242) 이미지를 통합시켜놓은 것. 3이 채널, 62* 242 짜리 이미지. 보통 채널이 뒤에 위치하기 때문에 옮겨야함
print(np.transpose(npimg, (1,2,0)).shape) # (62, 242, 3)

import matplotlib.pyplot as plt

# plt.figure(figsize=(10,7))
# plt.imshow(np.transpose(npimg, (1,2,0)))
# print(labels)
# plt.show()

one_img = images[1]
print(one_img.shape)
one_npimg = one_img.squeeze().numpy()
plt.title(f'"{labels[1]}" image')
plt.imshow(one_npimg, cmap="gray")
plt.show()
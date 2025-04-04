import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.utils import save_image
from PIL import Image, ImageFont, ImageDraw

num_epoch = 200
batch_size = 100
learing_rate = 0.0002
num_channel = 1
dir_name = 'CGAN_results'

noise_size = 100
hidden_size1 = 256
hidden_size2 = 512
hidden_size3 = 1024

condition_size = 10

import os
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(noise_size + condition_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 784)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        y = self.tanh(self.fc4(out))
        return y

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784 + condition_size, hidden_size3)
        self.fc2 = nn.Linear(hidden_size3, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size1)
        self.fc4 = nn.Linear(hidden_size1, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.leaky_relu(self.fc1(x))
        out = self.leaky_relu(self.fc2(out))
        out = self.leaky_relu(self.fc3(out))
        y = self.sigmoid(self.fc4(out))
        return y

import torchvision.datasets as dset
mnist_data = dset.MNIST(root='MNIST_data/',
                        train=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5)
                        ]),
                        download=True)

data_loader = DataLoader(dataset=mnist_data,
                         batch_size=batch_size,
                         shuffle=True)

discriminator = Discriminator()
generator = Generator()

import torch.optim as optim
d_optimizer = optim.Adam(discriminator.parameters(), lr=learing_rate)
g_optimizer = optim.Adam(generator.parameters(), lr=learing_rate)
loss_func = nn.BCELoss()

for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32)

        real_images = images.reshape(batch_size, -1)
        label_encoded = F.one_hot(label, num_classes=10)
        real_images_concat = torch.cat([real_images, label_encoded], dim=1)

        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        z = torch.randn(batch_size, noise_size)
        z_concat = torch.cat([z, label_encoded], dim=1)
        fake_images = generator(z_concat)
        fake_images_concat = torch.cat([fake_images, label_encoded], dim=1)

        g_loss = loss_func(discriminator(fake_images_concat), real_label)
        g_loss.backward()
        g_optimizer.step()


        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        z = torch.randn(batch_size, noise_size)
        z_concat = torch.cat([z, label_encoded], dim=1)
        fake_images = generator(z_concat)
        fake_images_concat = torch.cat([fake_images, label_encoded], dim=1)

        fake_loss = loss_func(discriminator(fake_images_concat), fake_label)
        real_loss = loss_func(discriminator(real_images_concat), real_label)
        d_loss = (fake_loss + real_loss) / 2

        d_loss.backward()
        d_optimizer.step()

        if (i+1) % 150 == 0:
            print('epoch:[{}/{}] step[{}/{}] d_loss:{:.5f} g_loss:{:.5f}'.format(
                epoch+1, num_epoch, i+1, len(data_loader), d_loss.item(), g_loss.item()
            ))

    samples = fake_images.reshape(batch_size, 1, 28, 28)
    save_image(samples, os.path.join(dir_name, 'CGAN_fake_samples{}.png'.format(epoch+1)))
    fake_sample_image = Image.open('{}/CGAN_fake_samples{}.png'.format(dir_name, epoch+1))
    font = ImageFont.truetype('arial.ttf', 17)

    label = label.tolist()
    label = label[:10]
    label = [str(l) for l in label]

    label_text = ', '.join(label)
    label_text = 'Contional GAN -\nfirst 10 label in this image:\n'+label_text

    image_edit = ImageDraw.Draw(fake_sample_image)
    image_edit.multiline_text(xy=(15,300),
                              text=label_text,
                              fill=(0,255,255),
                              font=font,
                              stroke_width=4,
                              stroke_fill=(0,0,0))
    fake_sample_image.save('{}/CGAN_fake_samples{}.png'.format(dir_name, epoch+1))

def check_condition(l_generator):
    test_image = torch.empty(0)

    for i in range(10):
        test_label = torch.tensor([0,1,2,3,4,5,6,7,8,9])
        test_label_encoded = F.one_hot(test_label, num_classes=10)

        z = torch.randn(10, noise_size)
        z_concat = torch.cat([z, test_label_encoded], dim=1)

        test_image = torch.cat([test_image, l_generator(z_concat)], dim=0)

    result = test_image.reshape(100, 1, 28, 28)
    save_image(result, os.path.join(dir_name, 'CGAN_test_result.png'), nrow=10)

check_condition(generator)



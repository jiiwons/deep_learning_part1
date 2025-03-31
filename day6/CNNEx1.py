import torch
import torch.nn as nn

inputs = torch.Tensor(1,1,28,28) # 초기화되지 않은 텐서를 생성. (배치크기, 채널수,이미지 높이, 이미지 너비)
print(inputs.shape)

conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1) #1은 same, 0은 valid
print(conv1)
# conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0, stride=1)
# print(conv1)

conv2 = nn.Conv2d(32, 64, 3, padding=1)
print(conv2)

pool = nn.MaxPool2d(kernel_size=2) # stride=2(기본값)
print(pool)
print()

output = conv1(inputs)
print(output.size()) #torch.Size([1, 32, 28, 28])

output = conv2(output)
print(output.size()) #torch.Size([1, 64, 28, 28])

output = pool(output)
print(output.size()) #torch.Size([1, 64, 14, 14])

output = output.view(output.size(0), -1)
print(output.size()) #torch.Size([1, 12544])

fclayter = nn.Linear(12544, 10)
output = fclayter(output)
print(output.size()) #torch.Size([1, 10])

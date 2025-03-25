import torch
import torch.nn as nn

x = torch.FloatTensor(torch.randn(16,10))

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        return y

CModel = CustomLinear(10,5)
y = CModel.forward(x)
print(y)
print()

y2 = CModel(x) # 위에 거랑 결과 같음. 모듈이 갖고 있는 __call__() 호출하고, forward 함수를 호출함/ 위 코드 말고 이런 식으로 써야함 
print(y2)

import torch
import torch.nn.functional as F

torch.manual_seed(777)

wsum = torch.randn(3, 5, requires_grad=True)
print(wsum)
hypothesis = F.softmax(wsum, dim=1)
print()
print(hypothesis)
print()

y = torch.randint(5,(3,)).long()
print(y)
print()

y_one_hot = torch.zeros_like(hypothesis)
print(y_one_hot)
print()

y_one_hot = y_one_hot.scatter(1, y.unsqueeze(dim=1),1) # unhsqueeze로 차원을 맞춤. y_one_hot을 참조해서 정답인 레이블에 1(세번째 매개변수)로 채움
print(y_one_hot)

print(-(y_one_hot * torch.log(F.softmax(wsum, dim=1))).sum(dim=1))
print(-(y_one_hot * torch.log(F.softmax(wsum, dim=1))).sum(dim=1).mean()) # 실질적으로 들어가는 loss값
print(-(y_one_hot * torch.log_softmax(wsum, dim=1)).sum(dim=1).mean())
print(F.cross_entropy(wsum,y)) # 소프트맥스 통과 안시켰음. 원핫인코딩 안시켰는데, cross_entropy내부에서 원핫인코딩 시킴/ 내부적으로 softmax + log + NLLLoss 과정을 수행


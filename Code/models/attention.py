import torch
from torch import nn

en_hn = torch.randn(60,256)
dc_hn = torch.randn(60,256)

cos = nn.CosineSimilarity(dim=1)
softmax = nn.LogSoftmax(dim=0)

similarity = cos(en_hn, dc_hn)
print(en_hn, dc_hn)
print(similarity)
print(similarity.shape)
print(softmax(similarity))
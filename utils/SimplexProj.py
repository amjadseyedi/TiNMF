import torch
from utils.SimplexColProj import SimplexColProj

def SimplexProj(y):
    x = torch.maximum(y, 0)
    if x.shape[0] == 1:
        x = torch.minimum(x, 1)
    else:
        K = torch.where(torch.sum(x, dim=0) > 1)[0]
        x[:, K] = SimplexColProj(y[:, K])
    return x


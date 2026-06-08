import torch

def SimplexColProj(Y, options):

    if options['dtype'] == torch.cuda.FloatTensor:
        ltype = torch.cuda.LongTensor
    else:
        ltype = torch.LongTensor
    Y = Y.T
    N, D = Y.shape
    X = torch.sort(Y, dim=1, descending=True).values
    Xtmp = (torch.cumsum(X, dim=1) - 1) @ torch.diag(1.0 / torch.arange(1, D+1).type(options['dtype']))
    X = torch.maximum(Y - Xtmp[torch.arange(N).type(ltype), torch.sum(X > Xtmp, axis=1)-1].reshape(-1, 1), torch.tensor(0))
    return X.T
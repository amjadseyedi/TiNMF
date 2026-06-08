import torch
import numpy as np
from utils.nnls_init import nnls_init
from utils.SimplexProj import SimplexProj
from utils.SimplexColProj import SimplexColProj

def nnls_FPGM(X, W, options=None):
    if options is None:
        options = {}
    
    options.setdefault('delta', 1e-6)
    options.setdefault('inneriter', 500)
    options.setdefault('proj', 0)
    options.setdefault('alpha0', 0.05)

    W = W
    m, n = X.shape
    m, r = W.shape
    WtW = W.T @ W
    WtX = W.T @ X

    if 'init' not in options or options['init'] is None:
        H = nnls_init(X, W, WtW, WtX, options).T
    else:
        H = options['init']

    L = torch.norm(WtW)
    alpha0 = options['alpha0']
    alpha = [alpha0]

    if options['proj'] == 1:
        H = SimplexProj(H)
    elif options['proj'] == 0:
        H = torch.maximum(H, torch.tensor(0))
    elif options['proj'] == 2:
        H = SimplexColProj(H.T, options).T
    elif options['proj'] == 3:
        H = SimplexColProj(H, options)

    Y = H
    i = 0
    eps0 = 0
    eps = 1

    while i < options['inneriter'] and eps >= options['delta'] * eps0:
        Hp = H
        alpha.append((np.sqrt(alpha[i]**4 + 4*alpha[i]**2) - alpha[i]**2) / 2)
        beta = alpha[i] * (1 - alpha[i]) / (alpha[i]**2 + alpha[i+1])

        H = Y - (WtW @ Y - WtX) / L

        if options['proj'] == 1:
            H = SimplexProj(H)
        elif options['proj'] == 0:
            H = torch.maximum(H, torch.tensor(0))
        elif options['proj'] == 2:
            H = SimplexColProj(H.T, options).T
        elif options['proj'] == 3:
            H = SimplexColProj(H, options)

        Y = H + beta * (H - Hp)

        if i == 0:
            eps0 = torch.norm(H - Hp)
        eps = torch.norm(H - Hp)
        i += 1

    return H, WtW, WtX
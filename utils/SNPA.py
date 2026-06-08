import torch
from utils.nnls_FPGM import nnls_FPGM


def SNPA(X: torch.Tensor, r: int, options: dict | None = None):
    """GPU-friendlier SNPA implementation.

    Assumptions:
      - X is a 2D torch.Tensor with shape (m, n).
      - If you want CUDA execution, pass X already on CUDA, e.g. X = X.cuda().
      - nnls_FPGM accepts CUDA tensors and returns H with shape (k, n).
    """
    if X.ndim != 2:
        raise ValueError("X must be a 2D tensor with shape (m, n).")
    if r < 0:
        raise ValueError("r must be non-negative.")

    m, n = X.shape
    r = min(int(r), n)

    opts = {} if options is None else dict(options)
    opts.setdefault("normalize", 0)
    opts.setdefault("display", 1)
    opts.setdefault("maxitn", 200)
    opts.setdefault("relerr", 1e-6)
    opts.setdefault("proj", 0)

    device = X.device
    dtype = X.dtype

    # GPU-safe column normalization. Avoid scipy/sparse diagonal matrices.
    if opts["normalize"] == 1:
        X = X / (torch.sum(X, dim=0, keepdim=True) + torch.finfo(dtype).eps)

    # Initialization.
    normX0 = torch.sum(X * X, dim=0)
    nXmax = torch.max(normX0)

    # Degenerate input: avoid division by zero and undefined H.
    if r == 0 or nXmax <= 0:
        W = X.new_empty((m, 0))
        H = X.new_empty((0, n))
        K = torch.empty((0,), dtype=torch.long, device=device)
        return W, H, K

    normR = normX0.clone()

    # Preallocate instead of repeatedly vstack/hstack-ing inside the loop.
    K = torch.empty((r,), dtype=torch.long, device=device)
    U = torch.empty((m, r), device=device, dtype=dtype)
    XtUK = torch.empty((r, n), device=device, dtype=dtype)
    UKtUK = torch.empty((r, r), device=device, dtype=dtype)

    H = X.new_empty((0, n))
    selected = 0

    for i in range(r):
        # This scalar check synchronizes once per iteration. That is unavoidable
        # because the Python loop termination depends on the value.
        rel_error = torch.sqrt(torch.max(normR) / nXmax)
        if rel_error.item() <= opts["relerr"]:
            break

        # Select the column of the residual R with largest l2 norm.
        b0 = torch.argmax(normR)
        denom = torch.clamp(normR[b0].abs(), min=torch.finfo(dtype).eps)

        # Check ties up to 1e-6 precision.
        tied = torch.nonzero((normR[b0] - normR) / denom <= 1e-6, as_tuple=False).flatten()

        # In case of a tie, select the tied column with largest original norm.
        if tied.numel() > 1:
            b = tied[torch.argmax(normX0[tied])]
        else:
            b = tied[0]

        K[i] = b
        U[:, i].copy_(X[:, b])

        # Update X^T U_K and U_K^T U_K using preallocated slices.
        XtUK[i] = X.T @ U[:, i]
        if i == 0:
            UKtUK[0, 0] = torch.dot(U[:, 0], U[:, 0])
        else:
            UtUi = U[:, :i].T @ U[:, i]
            UKtUK[:i, i] = UtUi
            UKtUK[i, :i] = UtUi
            UKtUK[i, i] = torch.dot(U[:, i], U[:, i])

        # Update residual via NNLS/FPGM.
        if i == 0:
            H, WtW, WtX = nnls_FPGM(X, X[:, K[: i + 1]], opts)
        else:
            H_init = torch.cat([H, X.new_zeros((1, n))], dim=0)
            H_init[i, K[i]] = 1
            opts["init"] = H_init
            H, WtW, WtX = nnls_FPGM(X, X[:, K[: i + 1]], opts)

        G = UKtUK[: i + 1, : i + 1]
        C = XtUK[: i + 1]
        normR = normX0 - 2 * torch.sum(C * H, dim=0) + torch.sum(H * (G @ H), dim=0)
        normR = torch.clamp(normR, min=0)

        selected = i + 1

    K = K[:selected]
    W = X[:, K]
    return W, H

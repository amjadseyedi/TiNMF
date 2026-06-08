import torch
from scipy import sparse
from scipy.linalg import pinv
from utils.orthNNLS import orthNNLS

def nnls_init(X, W, WtW, WtX, options):
    if torch.linalg.cond(W) > 1e6:  # Assign each column of X to the closest column of W
        # in terms of angle: this is the optimal solution with
        # V having a single non-zero per column.
        H = orthNNLS(X, W)
    else:  # Projected LS solution + scaling
        if sparse.issparse(X):
            H = torch.maximum(0, pinv(W) @ X)
        else:
            # H = np.maximum(0, np.linalg.solve(W, X))
            H, residuals, rank, s = torch.linalg.lstsq(X, W, rcond=None)
        
        # Scale
        alpha = torch.sum(H.T * WtX) / torch.sum(WtW * (H.T @ H))
        H = H * alpha

    # Check that no rows of H is zeros 
    # If it is the case, set them to small random numbers
    zerow = torch.where(torch.sum(H, dim=0) == 0)[0]
    H[zerow, :] = 0.001 * torch.max(H) * torch.rand(len(zerow), H.shape[1]).type(H.type())

    return H
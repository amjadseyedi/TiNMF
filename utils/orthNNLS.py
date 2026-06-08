import numpy as np

def orthNNLS(M, U, Mn=None):
    if Mn is None:
        norm2m = np.sqrt(np.sum(M**2, axis=0))
        Mn = M * (1.0 / (norm2m + 1e-16))
    
    m, n = Mn.shape
    m, r = U.shape
    
    # Normalize columns of U
    norm2u = np.sqrt(np.sum(U**2, axis=0))
    Un = U * (1.0 / (norm2u + 1e-16))
    
    A = Mn.T @ Un  # n by r matrix of angles between the columns of U and M
    b = np.argmax(A, axis=1)  # best column of U to approx. each column of M
    
    V = np.zeros((r, n))
    
    # Assign the optimal weights to V[b[i], i] > 0
    for i in range(n):
        V[b[i], i] = M[:, i].T @ U[:, b[i]] / np.linalg.norm(U[:, b[i]])**2
    
    norm2v = np.sqrt(np.sum(V**2, axis=0))
    
    return V, norm2v
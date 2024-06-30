import numpy as np


def estimate_norm(A1_matvec,A1_transpose_matvec,A2_matvec,A2_transpose_matvec,n,m=100):
    
    # todo: hutch++

    G = np.random.randn(n,m//3)
    Y1 = A1_matvec(G)-A2_matvec(G)
    Y2 = A1_transpose_matvec(Y1) - A2_transpose_matvec(Y1)
    Q,_ = np.linalg.qr(Y2)

    H = np.random.randn(n,m//3)
    Z = H - Q@(Q.T@H)

    t1 = np.linalg.norm(A1_matvec(Q) - A2_matvec(Q))**2
    t2 = np.linalg.norm(A1_matvec(Z) - A2_matvec(Z))**2 / (m//3)
    
    return np.sqrt(t1 + t2)
    
#    X = np.random.randn(n,m)
#    return np.linalg.norm(A1@X - A2.matvec(X))/np.sqrt(m)



def implicit_err(A1_matvec,A2_matvec,n,stride):

    err2 = 0.

    for i in range(n//stride):
        X = np.zeros((n,stride))
        X[i*stride:(i+1)*stride] = np.eye(stride)

        err2 += np.linalg.norm(A1_matvec(X)-A2_matvec(X))**2

    return np.sqrt(err2)    
    
def err_by_level(A,H,k):
    """
    """

    n,_ = A.shape
    E = np.zeros((n,n))

    q = int(np.ceil(np.log2(n/k))) # max levels
    err_sq = np.zeros(q+1)
    
    # iterate over levels
    for l in range(1,q+1):
        d = 2**l
        m = n//d

        errl = 0
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Aj = A[jp*m:(jp+1)*m,j*m:(j+1)*m]
            Hj = H[jp*m:(jp+1)*m,j*m:(j+1)*m]
            Ej = np.linalg.norm(Aj-Hj)
            E[jp*m:(jp+1)*m,j*m:(j+1)*m] = Ej
            err_sq[l-1] += Ej**2
            
    for j in range(d):

        Aj = A[j*m:(j+1)*m,j*m:(j+1)*m]
        Hj = H[j*m:(j+1)*m,j*m:(j+1)*m]
        Ej = np.linalg.norm(Aj - Hj)
        E[j*m:(j+1)*m,j*m:(j+1)*m] 
        err_sq[l] += Ej**2

    return E,err_sq
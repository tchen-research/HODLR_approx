import numpy as np
from .HODLR import *

def get_peel_sketch(n,d,s):
    
    m = n//d
    Ω = [np.zeros((n,s)),np.zeros((n,s))]

    for j in range(d):
        
        Ωj = np.random.randn(m,s)
        Ω[j%2][j*m:(j+1)*m] = Ωj
        
    return Ω

def RSVD_peel(A,k,s,truncate=True):
    """
    """
    
    n,_ = A.shape
    A_approx = HODLR(n)

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketch
        Ω = get_peel_sketch(n,d,s)
        AΩ = [A@Ω[i] - A_approx.matvec(Ω[i]) for i in [0,1]]

        Ψ = [np.zeros((n,min(s,m))),np.zeros((n,min(s,m)))]

        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m]
            Vj,_ = np.linalg.qr(Yj)
            Ψ[(j+1)%2][jp*m:(jp+1)*m] = Vj # build matrices for products with subpaces

        # simultaneously compute products with subspaces
        AΨ = [A.T@Ψ[i] - A_approx.transpose_matvec(Ψ[i]) for i in [0,1]]

        # get low-rank factors
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Vj = Ψ[(j+1)%2][jp*m:(jp+1)*m]
            Xj = AΨ[(j+1)%2][j*m:(j+1)*m].T

            if truncate:
                U,σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(σ[:k])@Vt[:k]
                
            A_approx.add_low_rank_factors(l,j,Vj,Xj)

    # recover diagonal blocks  
    Ψ = np.zeros((n,m))
    for j in range(d):
        Ψ[j*m:(j+1)*m] = np.eye(m)

    AΨ = A.T@Ψ - A_approx.transpose_matvec(Ψ)

    for j in range(d):

        Vj = Ψ[j*m:(j+1)*m]
        Xj = AΨ[j*m:(j+1)*m].T
        
        A_approx.add_diagonal_factors(l,j,Vj@Xj)
            
    return A_approx


def GN_peel(A,k,s,t,truncate=True):
    """
    """
    
    n,_ = A.shape
    A_approx = HODLR(n)

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketchs
        Ω = get_peel_sketch(n,d,s)        
        AΩ = [A@Ω[i] - A_approx.matvec(Ω[i]) for i in [0,1]]
        
        Ψ = get_peel_sketch(n,d,t)
        AΨ = [A.T@Ψ[i] - A_approx.transpose_matvec(Ψ[i]) for i in [0,1]]


        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m]
            Vj,_ = np.linalg.qr(Yj)

            Hj = Ψ[(j+1)%2][jp*m:(jp+1)*m]
            Zj = AΨ[(j+1)%2][j*m:(j+1)*m].T
            Xj = np.linalg.lstsq(Hj.T@Vj,Zj,rcond=None)[0]

            if truncate:
                U,σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(σ[:k])@Vt[:k]
            
            A_approx.add_low_rank_factors(l,j,Vj,Xj)
        
    # recover diagonal blocks    
    Ψ = np.random.randn(n,t)
    AΨ = A.T@Ψ - A_approx.transpose_matvec(Ψ)

    for j in range(d):
        jp = j+1 if j%2==0 else j-1

        Hj = Ψ[j*m:(j+1)*m]
        Zj = AΨ[j*m:(j+1)*m].T
        Xj = np.linalg.lstsq(Hj.T,Zj,rcond=None)[0]
        
        A_approx.add_diagonal_factors(l,j,Xj)

        
    return A_approx


def RSVD_peel_dense(A,k,s,truncate=True):
    """
    """
    
    n,_ = A.shape
    A_approx = np.zeros((n,n))

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketch
        Ω = get_peel_sketch(n,d,s)
        AΩ = A@Ω - A_approx@Ω

        Ψ = [np.zeros((n,min(s,m))),np.zeros((n,min(s,m)))]

        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m]
            Vj,_,_ = np.linalg.svd(Yj,full_matrices=False)
            Ψ[(j+1)%2][jp*m:(jp+1)*m] = Vj # build matrices for products with subpaces

        # simultaneously compute products with subspaces
        AΨ = A.T@Ψ - A_approx.T@Ψ

        # get low-rank factors
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Vj = Ψ[(j+1)%2][jp*m:(jp+1)*m]
            Xj = AΨ[(j+1)%2,j*m:(j+1)*m].T

            if truncate:
                U,σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(σ[:k])@Vt[:k]
                
            A_approx[jp*m:(jp+1)*m,j*m:(j+1)*m] = Vj@Xj

        if m <= k:
            break

    # recover diagonal blocks
    Ω = np.random.randn(n,m)
    AΩ = A@Ω - A_approx@Ω
    
    Ψ = np.zeros((n,m))
    
    for j in range(d):

        Yj = AΩ[j*m:(j+1)*m]
        Vj,_ = np.linalg.qr(Yj)
        Ψ[j*m:(j+1)*m] = Vj

    AΨ = A.T@Ψ - A_approx.T@Ψ

    for j in range(d):

        Vj = Ψ[j*m:(j+1)*m]
        Xj = AΨ[j*m:(j+1)*m].T
        
        A_approx[j*m:(j+1)*m,j*m:(j+1)*m] = Vj@Xj
            
    return A_approx


def GN_peel_dense(A,k,s,t,truncate=True):
    """
    """
    
    n,_ = A.shape
    A_approx = np.zeros((n,n))

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketchs
        Ω = get_peel_sketch(n,d,s)        
        AΩ = A@Ω - A_approx@Ω
        
        Ψ = get_peel_sketch(n,d,t)
        AΨ = A.T@Ψ - A_approx.T@Ψ

        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m]
            Vj,_ = np.linalg.qr(Yj)

            Hj = Ψ[(j+1)%2][jp*m:(jp+1)*m]
            Zj = AΨ[(j+1)%2,j*m:(j+1)*m].T
            Xj = np.linalg.lstsq(Hj.T@Vj,Zj,rcond=None)[0]

            if truncate:
                U,σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(σ[:k])@Vt[:k]
            
            A_approx[jp*m:(jp+1)*m,j*m:(j+1)*m] = Vj@Xj

        # print(Vj.shape,Xj.shape)

        if m <= k:
            break
        
    # recover diagonal blocks    
    Ψ = np.random.randn(n,t)
    AΨ = A.T@Ψ - A_approx.T@Ψ

    for j in range(d):
        jp = j+1 if j%2==0 else j-1

        Hj = Ψ[j*m:(j+1)*m]
        Zj = AΨ[j*m:(j+1)*m].T
        Xj = np.linalg.lstsq(Hj.T,Zj,rcond=None)[0]
        
        A_approx[j*m:(j+1)*m,j*m:(j+1)*m] = Xj
def err(A1,A2,n,stride):

    err2 = 0.

    for i in range(n//stride):
        X = np.zeros((n,stride))
        X[i*stride:(i+1)*stride] = np.eye(stride)

        err2 += np.linalg.norm(A1@X-A2.matvec(X))**2

    return np.sqrt(err2)    
        
    return A_approx



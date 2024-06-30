import numpy as np
from .HODLR import *


def opt_HODLR(A,r):
    """
    compute OPT for rank r HODLR
    """

    n,_ = A.shape
    if n<=r:
        return A

    # get best rank k approx of top right block
    A12 = A[:n//2,n//2:]
    U,s,Vt = np.linalg.svd(A12)
    A12r = U[:,:r]@np.diag(s[:r])@Vt[:r]
    
    # get best rank k approx of bottom left block
    A21 = A[n//2:,:n//2]
    U,s,Vt = np.linalg.svd(A21)
    A21r = U[:,:r]@np.diag(s[:r])@Vt[:r]

    A11 = A[:n//2,:n//2]
    A22 = A[n//2:,n//2:]

    # recurse
    X = np.block([[opt_HODLR(A11,r),A12r],
                  [A21r,opt_HODLR(A22,r)]])

    return X

def fs_prod(X,Y):
    p,t = np.shape(X)
    pu,v = np.shape(Y)

    assert pu%p ==0, 'first dimension of X must divide first dimension of Y'

    u = pu//p

    Z = np.zeros((pu,t*v))
    
    for j in range(p):
        Z[u*j:u*(j+1)] = np.kron(X[j,:],Y[u*j:u*(j+1)])

    return Z

def count_sketch(d,t):

    ξ = np.zeros((d,t))
    ρ = np.random.choice(t,size=(d))
    
    for j in range(d):
        ξ[j,ρ[j]] = 1

    return ξ,ρ

def perforated_count_sketch(d,t):

    ξ,ρ = count_sketch(d,t)

    xp = np.kron(np.ones(d//2),[1,0]).reshape(d,1)
    xm = np.kron(np.ones(d//2),[0,1]).reshape(d,1)
    ξp = fs_prod(xp,ξ)
    ξm = fs_prod(xm,ξ)

    return [ξp,ξm],ρ

def GN_perf_peel(A_matvec,A_transpose_matvec,n,k,sr,tr,sl,tl,truncate=True):
    """
    """
    
    A_approx = HODLR(n)

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketchs
        ξ,ρ = perforated_count_sketch(d,tr)
        Ω_ = np.random.randn(n,sr)
        Ω = [fs_prod(ξ[i],Ω_) for i in [0,1]]
        AΩ = [A_matvec(Ω[i]) - A_approx.matvec(Ω[i]) for i in [0,1]]
        
        ζ,σ = perforated_count_sketch(d,tl)
        Ψ_ = np.random.randn(n,sl)
        Ψ = [fs_prod(ζ[i],Ψ_) for i in [0,1]]
        AΨ = [A_transpose_matvec(Ψ[i]) - A_approx.transpose_matvec(Ψ[i]) for i in [0,1]]


        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m,ρ[j]*sr:(ρ[j]+1)*sr]
            
            Vj,_ = np.linalg.qr(Yj)

            Hj = Ψ_[jp*m:(jp+1)*m]
            Zj = AΨ[(j+1)%2][j*m:(j+1)*m,σ[jp]*sl:(σ[jp]+1)*sl].T
            Xj = np.linalg.lstsq(Hj.T@Vj,Zj,rcond=None)[0]

            if truncate:
                U,Σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(Σ[:k])@Vt[:k]
            
            A_approx.add_low_rank_factors(l,j,Vj,Xj)
        
    # recover diagonal blocks    
    ζ,σ = count_sketch(d,tl)
    Ψ_ = np.random.randn(n,sl)
    Ψ = fs_prod(ζ,Ψ_) 
    AΨ = A_transpose_matvec(Ψ) - A_approx.transpose_matvec(Ψ)

    for j in range(d):

        Hj = Ψ_[j*m:(j+1)*m]
        Zj = AΨ[j*m:(j+1)*m,σ[j]*sl:(σ[j]+1)*sl].T
        Xj = np.linalg.lstsq(Hj.T,Zj,rcond=None)[0]
        
        A_approx.add_diagonal_factors(l,j,Xj)

        
    return A_approx

def RSVD_perf_peel(A_matvec,A_transpose_matvec,n,k,sr,tr,tl,truncate=True):
    """
    """
    
    A_approx = HODLR(n)

    q = int(np.ceil(np.log2(n/k))) # max levels

    # iterate over levels
    for l in range(1,q+1):

        d = 2**l
        m = n//d

        # Gaussian sketchs
        ξ,ρ = perforated_count_sketch(d,tr)
        Ω_ = np.random.randn(n,sr)
        Ω = [fs_prod(ξ[i],Ω_) for i in [0,1]]
        AΩ = [A_matvec(Ω[i]) - A_approx.matvec(Ω[i]) for i in [0,1]]

        Ψ_ = np.zeros((n,min(sr,m)))

        # get subspaces
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            Yj = AΩ[j%2][jp*m:(jp+1)*m,ρ[j]*sr:(ρ[j]+1)*sr]
            Vj,_ = np.linalg.qr(Yj)
            Ψ_[jp*m:(jp+1)*m] = Vj
     
        ζ,σ = perforated_count_sketch(d,tl)
        Ψ = [fs_prod(ζ[i],Ψ_) for i in [0,1]]
        AΨ = [A_transpose_matvec(Ψ[i]) - A_approx.transpose_matvec(Ψ[i]) for i in [0,1]]

        # get low-rank factors
        for j in range(d):
            jp = j+1 if j%2==0 else j-1       

            Vj = Ψ_[jp*m:(jp+1)*m]
            Xj = AΨ[(j+1)%2][j*m:(j+1)*m,σ[jp]*min(sr,m):(σ[jp]+1)*min(sr,m)].T
            
            if truncate:
                U,Σ,Vt = np.linalg.svd(Xj,full_matrices=False)
                Xj = U[:,:k]@np.diag(Σ[:k])@Vt[:k]
            
            A_approx.add_low_rank_factors(l,j,Vj,Xj)
        
    # recover diagonal blocks    
    Ψ_ = np.zeros((n,m))
    for j in range(d):
        Ψ_[j*m:(j+1)*m] = np.eye(m)

    ζ,σ = count_sketch(d,tl)
    Ψ = fs_prod(ζ,Ψ_) 
    AΨ = A_transpose_matvec(Ψ) - A_approx.transpose_matvec(Ψ)

    for j in range(d):

        Xj = AΨ[j*m:(j+1)*m,σ[j]*m:(σ[j]+1)*m].T
        
        A_approx.add_diagonal_factors(l,j,Xj)

        
    return A_approx
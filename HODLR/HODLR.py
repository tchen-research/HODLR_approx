import numpy as np

class HODLR:

    def __init__(self,n):

        self.n = n
        self.q = 0 
        self.V = {}
        self.X = {}
        self.D = {}

    def add_low_rank_factors(self,l,j,V,X):

        self.q = max(self.q,l)
        self.V[l,j] = V
        self.X[l,j] = X

    def add_diagonal_factors(self,l,j,D):
        
        self.q = l
        self.D[l,j] = D

    def matvec(self,x):
        y = np.zeros_like(x)
        if self.q == 0:
            return y
            
        for l in range(1,self.q+1):
            
            d = 2**l
            m = self.n//d
            
            for j in range(d):
                jp = j+1 if j%2==0 else j-1
                y[jp*m:(jp+1)*m] += self.V[l,j]@(self.X[l,j]@x[j*m:(j+1)*m])
                    
        for j in range(d):
            try:
                y[j*m:(j+1)*m] += self.D[l,j]@x[j*m:(j+1)*m]
            except:
                pass
        return y

    def transpose_matvec(self,x):

        y = np.zeros_like(x)
        if self.q == 0:
            return y
            
        for l in range(1,self.q+1):
            
            d = 2**l
            m = self.n//d
            
            for j in range(d):
                jp = j+1 if j%2==0 else j-1
                y[j*m:(j+1)*m] += self.X[l,j].T@(self.V[l,j].T@x[jp*m:(jp+1)*m])

        for j in range(d):
            try:
                y[j*m:(j+1)*m] += self.D[l,j].T@x[j*m:(j+1)*m]
            except:
                pass
        return y


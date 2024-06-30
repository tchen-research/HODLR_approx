
def get_sketch(s,t,n,d):

    ξ = [np.zeros((d,t)),np.zeros((d,t))]
    for j in range(d):
        σ = np.random.randint(t)
        if j%2==0:
            ξ[0][j,σ] = 1
        else:
            ξ[1][j,σ]=1

    G = np.random.randn(n,s*t)
    S = [np.kron(ξ[0],np.ones((n//d,s))) * G,\
         np.kron(ξ[1],np.ones((n//d,s))) * G]

    return ξ,S
    
def count_peel(A,s,t):

    n,_ = A.shape
    A_approx = np.zeros((n,n))

    for l in range(1,int(np.log2(n))):

        d = 2**l

        # form sketch of A
        ξ,S = get_sketch(s,t,n,d)
        AS = A@S - A_approx@S

        # form sketch of A^T
        s1 = 2*s
        ζ,T = get_sketch(s1,1,n,d)
        TtA = np.transpose(A.T@T,axes=(0,2,1)) - np.transpose(A_approx.T@T,axes=(0,2,1))

        # iterate over rows
        for j in range(d):
            jp = j+1 if j%2==0 else j-1

            # get approximate range
            σ = np.nonzero(ξ[j%2][j])[0][0]

            Yj = AS[j%2][jp*(n//d):(jp+1)*(n//d),σ*s:(σ+1)*s]
            Vj,_ = np.linalg.qr(Yj)

            # get low-rank approx
            ρ = np.nonzero(ζ[jp%2][jp])[0][0]

            Zj = TtA[jp%2][ρ*s1:(ρ+1)*s1,j*(n//d):(j+1)*(n//d)]
            Hj = T[jp%2][jp*(n//d):(jp+1)*(n//d),ρ*s1:(ρ+1)*s1]
            Xj = np.linalg.lstsq(Hj.T@Vj,Zj,rcond=None)[0]

            A_approx[jp*(n//d):(jp+1)*(n//d),j*(n//d):(j+1)*(n//d)] = Vj@Xj

        # if next iteration would produce too small of blocks, then just break and find diagonal blocks
        if n//d <= s:
            break


    # form sketch of A
    ξ,S = get_sketch(s,t,n,d)
    ξ = ξ[0] + ξ[1]
    S = S[0] + S[1]
    AS = A@S - A_approx@S

    # form sketch of A^T
    ζ,T = get_sketch(s,t,n,d)
    ζ = ζ[0] + ζ[1]
    T = T[0] + T[1]
    TtA = np.transpose(A.T@T) - np.transpose(A_approx.T@T)

    # iterate over rows
    for j in range(d):

        # get approximate range
        σ = np.nonzero(ξ[j])[0][0]

        Yj = AS[j*(n//d):(j+1)*(n//d),σ*s:(σ+1)*s]
        Vj,_ = np.linalg.qr(Yj)

        # get low-rank approx
        ρ = np.nonzero(ζ[j])[0][0]

        Zj = TtA[ρ*s:(ρ+1)*s,j*(n//d):(j+1)*(n//d)]
        Hj = T[j*(n//d):(j+1)*(n//d),ρ*s:(ρ+1)*s]
        Xj = np.linalg.lstsq(Hj.T@Vj,Zj,rcond=None)[0]

        A_approx[j*(n//d):(j+1)*(n//d),j*(n//d):(j+1)*(n//d)] = Vj@Xj

    return A_approx
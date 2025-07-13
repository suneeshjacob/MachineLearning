def svm(X,y,kernel='linear', support_vectors = [0,1]):

    if kernel == 'linear':
        phi = lambda x: x
    else:
        phi = kernel
    
    sv = [(X[i],y[i]) for i in support_vectors]
    N = len(sv)
    A = np.zeros((N+1,N+1))
    b = np.ones(N+1)

    for i in range(N):
        xi = sv[i][0]
        yi = sv[i][1]
        phi_i = phi(xi)
        
        for j in range(N):
            xj = sv[j][0]
            yj = sv[j][1]
            phi_j = phi(xj)
            A[i,j] = yi*yj*np.dot(phi_i,phi_j)
        
        A[i,N] = yi
    
    for j in range(N):
        yj = sv[j][1]
        A[N,j] = yj
    
    A[N,N] = 0
    b[N] = 0

    alpha = np.linalg.solve(A,b)

    weights = np.zeros(len(phi_i))
    for i in range(N):
        xi = sv[i][0]
        yi = sv[i][1]
        phi_i = phi(xi)
        weights += alpha[i]*yi*phi_i

    bias = alpha[-1]

    return weights, bias, alpha

X = np.matrix('2 2;2 -2;-2 -2;-2 2;1 1;1 -1;-1 -1;-1 1').A
y = np.matrix('1 1 1 1 -1 -1 -1 -1').A[0]
kernel = lambda x:np.array([x[0],x[1],(x[0]**2+x[1]**2-5)/3])
support_vectors=[0,1,2,3,6,7]
svm(X,y,support_vectors=support_vectors,kernel=kernel)

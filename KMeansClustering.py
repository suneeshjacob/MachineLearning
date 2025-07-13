def k_means_clustering(X,C):
    C = np.array(C)
    m = X.shape[0]
    k = len(C)
    while True:
        d = np.array([np.linalg.norm(X-i,axis=1) for i in C]).T
        y = np.argmin(d,axis=1)
        Xy = np.c_[X,y]
        C_new = [None for i2 in range(k)]
        for j in range(k):
            X_current_class = X[y==j,:]
            if len(X_current_clas)>0:
                C_new[j] = np.mean(X_current_class,axis=0)
            else:
                C_new[j] = C[j]
        C_new = np.array(C_new)
        print('\n','Centroid:', C, '\n\n', np.c_[X,d,y].round(2))
        if (C_new == C).all():
            return Xy
        C = C_new

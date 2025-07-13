def F_statistic(X,y):
    classes = np.unique(y)
    k = len(classes)
    n = len(X)
    c = X.shape[1]
    mean = X.mean(axis=0)
    ssw = 0
    ssb = np.zeros(c)
    for i in classes:
        X_i = X[np.where(y==i)]
        mean_i = X_i.mean(axis=0)
        ssw_i = np.square(X_i-mean_i).sum(axis=0)
        ssw += ssw_i
        n_i = len(X_i)
        ssb_i = n_i*np.square(mean_i - mean)
        #print(ssb_i)
        ssb += ssb_i
        
    dof_ssw = n-k
    dof_ssb = k-1
    msw = ssw/dof_ssw
    msb = ssb/dof_ssb
    F = np.divide(msb, msw)
    return F

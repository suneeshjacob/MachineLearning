A = np.array([[ 0.15,  0.18,  0.25, -0.5 ,  0.45, -0.04, -0.73, -0.05],
       [-0.21, -0.51, -0.65, -0.05,  0.29, -0.  ,  0.06,  0.28],
       [-0.24,  0.51, -0.53, -0.15,  0.08, -0.37, -0.25, -0.45],
       [-0.7 , -0.12, -0.22,  0.08, -0.03, -0.1 , -0.28, -0.16],
       [-0.21,  0.06,  0.32,  0.16,  0.27,  0.  ,  0.01, -0.22],
       [ 0.16,  0.29, -0.09,  0.67, -0.75,  0.37,  0.11,  0.  ],
       [ 0.08,  0.2 , -0.36, -0.51, -0.19, -0.89, -0.06, -0.19],
       [ 0.29, -0.26,  0.87,  0.5 ,  0.04, -0.49,  0.04,  0.24]])

W = np.array([[ 0.37, 0.33, -0.76],
                [-0.52, -0.65, 0.19],
                [-0.44, -0.03, -0.81]])

b=0.89

def same_padding(B):
    C = B.copy()
    C = np.c_[C[:,:1],C,C[:,-1:]]
    C = np.r_[C[:1,:],C,C[-1:,:]]
    return C

def convolution(image,kernel,bias,activation=sigmoid,stride=1,padding=None):
    A = image.copy()
    b = bias
    kernel_dim = kernel.shape
    n = image.shape[0]
    k = kernel_dim[0]
    if padding == None:
        p = 0
    f = int((n-k+2*p)/stride+1)
    z = np.zeros((f,f))
    for i in range(f):
        for j in range(f):
            ii = np.arange(i,i+kernel_dim[0])
            jj = np.arange(j,j+kernel_dim[0])
            x2 = A[ii,:][:,jj].flatten()
            x1 = W.flatten()
            s = []
            for i1 in range(len(x1)):
                s_temp = ''
                if x1[i1]<0:
                    s_temp+=f'({x1[i1]})'
                else:
                    s_temp+=f'{x1[i1]}'
                s_temp+=f' x '
                if x2[i1]<0:
                    s_temp+=f'-({-x2[i1]})'
                else:
                    s_temp+=f'{x2[i1]}'
                s.append(s_temp)
            
            
            z_current = activation(np.multiply(A[ii,:][:,jj],W).sum()+b)
            z[i,j]=z_current
    return z

sigmoid = lambda x: 1/(1+np.exp(-x))

z = np.zeros((6,6))
for i in range(6):
    for j in range(6):
        ii = np.arange(i,i+3)
        jj = np.arange(j,j+3)
        #ii = [0,1,2]
        #jj = [0,1,2]
        x2 = A[ii,:][:,jj].flatten()
        x1 = W.flatten()
        s = []
        for i1 in range(len(x1)):
            #s+=f'{x1[i1]} x {x2[i1]} = {round(x1[i1]*x2[i1],2)}'
            s_temp = ''
            if x1[i1]<0:
                s_temp+=f'({x1[i1]})'
            else:
                s_temp+=f'{x1[i1]}'
            s_temp+=f' x '
            if x2[i1]<0:
                s_temp+=f'-({-x2[i1]})'
            else:
                s_temp+=f'{x2[i1]}'
            s.append(s_temp)
        z_current = sigmoid(np.multiply(A[ii,:][:,jj],W).sum()+b).round(2)
        z[i,j]=z_current
    #print(' + '.join(s))

vocab = ['am', 'are', 'here', 'how', 'i', 'name', 'where', 'you']

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Numerical stability
    return e_x / e_x.sum(axis=0)

def onehotencoding(string):
    words = string.split(' ')
    A = np.zeros((len(words),len(vocab)))
    for i in range(len(words)):
        A[i,:] = words[i]==np.array(vocab)
    return A

def onehotdecoding(A):
    string = ' '.join([vocab[np.where(A[i,:]==1)[0][0]] for i in range(A.shape[0])])
    return string


Wxh = [[-0.49, -0.74, -0.96, -1.23, -0.44, 1.61, -1.32, 0.01], [-0.06, 1.24, -1.18, 0.41, -1.83, -1.54, -0.41, -0.57]]
Whh = [[-3.78, 6.39], [-6.25, 1.71]]
Why = [[2.91, 6.41], [0.08, 1.16], [5.36, -3.93], [-0.18, -0.4], [-5.55, 1.38], [-0.49, -0.64], [-0.81, -0.14], [-2.21, -0.18]]
bh = [[-1.25], [0.81]]
by = [[-1.26], [-2.15], [1.97], [-1.35], [6.0], [-1.16], [-1.22], [-0.84]]




x1, x2, x3 = onehotencoding('where are you')
h0 = np.array([0,0])

x1 = x1.reshape(-1,1)
x2 = x2.reshape(-1,1)
x3 = x3.reshape(-1,1)
h0 = h0.reshape(-1,1)


h1 = sigmoid(Wxh@x1 + Whh@h0 + bh)
y1 = softmax(Why@h1 + by)

h2 = sigmoid(Wxh@x2 + Whh@h1 + bh)
y2 = softmax(Why@h2 + by)

h3 = sigmoid(Wxh@x3 + Whh@h2 + bh)
y3 = softmax(Why@h3 + by)

' '.join(np.array(vocab)[np.argmax(np.array([y1,y2,y3]),axis=1)].T[0].tolist())

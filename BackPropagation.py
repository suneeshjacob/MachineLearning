def sigmoid(x): return 1/(1+np.exp(-x))
def dsigmoid(x): return sigmoid(x)*(1-sigmoid(x))

w1 = np.matrix('1 2 3;4 5 6; 7 8 9 ;10 11 12').A/10
w2 = np.matrix('1 2 3 4;5 6 7 8;9 10 11 12;13 14 15 16').A/10
w3 = np.matrix('1 2 3 4;5 6 7 8').A/10
b1 = np.matrix('1;2;3;4').A/10
b2 = np.matrix('1;2;3;4').A/10
b3 = np.matrix('1;2').A/10
x = np.matrix('5;10;15').A/10
y = np.matrix('5;10')

x1 = x.copy()

z2=w1@x1+b1
x2 = sigmoid(z2)
z3=w2@x2+b2
x3 = sigmoid(z3)
z4 =w3@x3+b3
x4 = sigmoid(z4)

def flatten_weights(weights):
    flattened_weights = []
    structure = [len(weights[0][0])]
    for i in range(len(weights)):
        flattened_weights += weights[i].flatten().tolist()
        if i%2==0:
            structure.append(len(weights[i]))
    flattened_weights = np.array(flattened_weights)
    return flattened_weights, structure
    
def unflatten_weights(flattened_weights, structure):
    unflattened_weights = []
    cursor = 0
    for i in range(len(structure)-1):
        k = flattened_weights[cursor:cursor+structure[i]*structure[i+1]]
        k.resize(structure[i+1],structure[i])
        unflattened_weights.append(k)
        cursor += structure[i]*structure[i+1]
        k = flattened_weights[cursor:cursor+structure[i+1]]
        k.resize(structure[i+1],1)
        unflattened_weights.append(k)
        cursor += structure[i+1]
    return unflattened_weights


def loss(weights, inputs, outputs):
    x = inputs.copy()
    n = int(len(weights)/2)
    for i in range(n):
        w = weights[2*i]
        b = weights[2*i+1]
        x = sigmoid(w@x+b)
    
    return 0.5*np.mean(np.square(outputs-x))



inputs = x1
outputs = y

def f(x, structure = [3,4,4,2]):
    unflattened_weights = unflatten_weights(x, structure)
    l = loss(unflattened_weights, inputs, outputs)
    return l

def gradient(f,x,h=0.001):
    n = len(x)
    grad = []
    for i in range(n):
        xph = []
        for j in range(n):
            if j == i:
                xph.append(x[j]+h)
            else:
                xph.append(x[j])
        xph = np.array(xph)
        grad.append((f(xph)-f(x))/h)
    return np.array(grad)

weights = [w1,b1,w2,b2,w3,b3]
structure = [3,4,4,2]

kk = unflatten_weights(gradient(f,flatten_weights(weights)[0],h=0.000001),structure)

d4 = (x4-y)/2
d3 = (w3.T)@ np.multiply(d4,dsigmoid(z4))
d2 = (w2.T)@ np.multiply(d3,dsigmoid(z3))
d1 = (w1.T)@ np.multiply(d2,dsigmoid(z2))

d4 = d4.A
d3 = d3.A
d2 = d2.A
d1 = d1.A

db3 = np.multiply(d4,dsigmoid(z4))
db2 = np.multiply(d3,dsigmoid(z3))
db1 = np.multiply(d2,dsigmoid(z2))

# dw3 = np.array([db3@i for i in x3]).T
# dw2 = np.array([db2@i for i in x2]).T
# dw1 = np.array([db1@i for i in x1]).T

dw3 = db3@x3.T
dw2 = db2@x2.T
dw1 = db1@x1.T

def forward_propagation(x,y,weights):

    layer_values = [x]
    z_values = []

    n = int(len(weights)/2)

    for i in range(n):
        w = weights[2*i]
        b = weights[2*i+1]
        z_values.append(w@layer_values[-1]+b)
        layer_values.append(sigmoid(z_values[-1]))

    return layer_values, z_values

def back_propagation(x,y,weights_list):
    weights = weights_list.copy()
    layer_values, z_values = forward_propagation(x,y,weights)
    n = int(len(weights)/2)

    gradient_of_weights = []

    a = layer_values.pop(-1)
    d = (a-y)/len(y)

    for _ in range(n):
        z = z_values.pop(-1)
        db = np.multiply(d,dsigmoid(z))
        gradient_of_weights.append(db)
        a = layer_values.pop(-1)
        dw = db@a.T
        gradient_of_weights.append(dw)
        _ = weights.pop(-1)
        w = weights.pop(-1)
        d = w.T@db


    # d4 = 0.5*(x4-y)
    # db3 = np.multiply(d4,dsigmoid(z4))
    # dw3 = db3@x3.T
    # d3 = w3.T@db3

    # db2 = np.multiply(d3,dsigmoid(z3))
    # dw2 = db2@x2.T
    # d2 = w2.T@db2

    # db1 = np.multiply(d2,dsigmoid(z2))
    # dw1 = db1@x1.T
    # d1 = w1.T@db1
    return gradient_of_weights[::-1]

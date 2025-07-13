import numpy
import numpy as np

dataset = [['1', 'Sunny', 'Hot', 'High', 'Weak', 'No'],
 ['2', 'Sunny', 'Hot', 'High', 'Strong', 'No'],
 ['3', 'Overcast', 'Hot', 'High', 'Weak', 'Yes'],
 ['4', 'Rainy', 'Mild', 'High', 'Weak', 'Yes'],
 ['5', 'Rainy', 'Cool', 'Normal', 'Weak', 'Yes'],
 ['6', 'Rainy', 'Cool', 'Normal', 'Strong', 'No'],
 ['7', 'Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
 ['8', 'Sunny', 'Mild', 'High', 'Weak', 'No'],
 ['9', 'Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
 ['10', 'Rainy', 'Mild', 'Normal', 'Weak', 'Yes'],
 ['11', 'Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
 ['12', 'Overcast', 'Mild', 'High', 'Strong', 'Yes'],
 ['13', 'Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
 ['14', 'Rainy', 'Mild', 'High', 'Strong', 'No']]

def IG_calc(data, feature):
    #feature = 4
    all_classes = list(set(data[:,-1]))
    p_beforesplit = [numpy.sum(data[:,-1]==i) for i in all_classes]
    s_beforesplit = sum(p_beforesplit)
    E_beforesplit = sum([-(i/s_beforesplit)*numpy.log2(i/s_beforesplit) if (i/s_beforesplit)!=0 else 0 for i in p_beforesplit])
    n = len(data[:,feature])
    E_eff = 0
    for j in set(data[:,feature]):
        k = sum(data[:,feature]==j)
        d = data[data[:,feature]==j]
        p = [numpy.sum(d[:,-1]==i) for i in all_classes]
        s = sum(p)
        E = sum([-(i/s)*numpy.log2(i/s) if (i/s)!=0 else 0 for i in p])
        E_eff += (k/n)*E
    IG = E_beforesplit - E_eff
    #print(IG)
    return IG


# DT - Classifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import graphviz
import matplotlib.pyplot as plt

X = np.matrix('2 3;1 5;5 1;6 2;4 4').reshape(-1, 2).A
y = np.array([0,0,0,1,1])

tree = DecisionTreeClassifier(criterion='entropy')  
tree.fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(tree, 
          feature_names=['$x_1$', '$x_2$'], 
          class_names=['Class 0', 'Class 1'], 
          filled=True, 
          rounded=True)
plt.show()

# DT - Regressor
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

X = np.array([[2, 3], [1, 5], [5, 1], [6, 2], [4, 4]])
y = np.array([0.1, 0.2, 0.3, 0.7, 0.9])

tree = DecisionTreeRegressor(criterion='squared_error')
tree.fit(X, y)

plt.figure(figsize=(10, 6))
plot_tree(tree, 
          feature_names=['$x_1$', '$x_2$'], 
          filled=True, 
          rounded=True)
plt.show()


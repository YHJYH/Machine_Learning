### Part 2

# import libraries
import numpy as np
from matplotlib import pyplot as plt

# Spectral Clustering Algorithm
def weight(X, c):
    # Calculate Adjacency weight matrix W.
    l,n = X.shape
    W = np.ones((l,l))
    for i in range(l):
        for j in range(l):
            W[i,j] = np.exp(-c * (np.linalg.norm(X[i]-X[j]))**2)
    return W
            
def laplacian(W):
    # Calculate Graph Laplacian L.
    l,l = W.shape
    D = np.zeros((l,l))
    for i in range(l):
        D[i,i] = np.sum(W[i])
    L = D - W
    return L

def sec_eig(L):
    # Find the second smallest eigenvector of L.
    val, vec = np.linalg.eig(L)
    a = np.argsort(val)[1]
    return vec[:,a]

def sign(x):
    # Return the sign of x.
    if x >= 0:
        y = 1
    else:
        y = -1
    return y

def spectral_cluster(X, c):
    # Use data X and c to give weight W, laplacian L, 
    # cluster vector v2, and use them to do the clustering.
    W = weight(X, c)
    L = laplacian(W)
    v2 = sec_eig(L)
    y = np.ones(v2.shape)
    for i in range(len(v2)):
        y[i] = sign(v2[i])
    return y

def best_cluster(X, Y, c):
    # Implement spectral_cluster function with different c 
    # to determine the best c which gives the best cluster.
    # The return value is the best c, corresponding correctness and prediction.
    l,n = X.shape
    final_y = []
    final_c = 0
    final_cor = 0
    for i in c:
        y = spectral_cluster(X, i)
        cor = max((y == Y).sum(), (-y == Y).sum())
        if final_cor < cor:
            final_cor = cor
            final_c = i
            final_y = y
    final_cor_rate = final_cor / l
    
    return final_c, final_cor_rate, final_y

### Experiment 1

# Load data
twomoons = np.loadtxt('C:/Users/james007/Desktop/twomoons.dat.txt')
X_moons = twomoons[:, 1:]
Y_moons = twomoons[:, 0]

# Give a sequence of c and find the best cluster of data
c1 = np.linspace(0, 100, 101)[1:]
moons_c, moons_cor_rate, moons_y = best_cluster(X_moons, Y_moons, c1)
print(f'The best c is {moons_c} and it gives a correctness of{moons_cor_rate: .2%}')

# Plot of the comparision
plt.figure(figsize=(12,6))

# Plot of original data
plt.subplot(121)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c = Y_moons)
plt.title('Original data')

# Plot of spectral clustering
plt.subplot(122)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c = moons_y)
plt.title('Spectral clustering algorithm')

plt.savefig('P2E1')

### Experiment 2
I = np.eye(2)

# Generate random class '-1'
x_1 = np.random.multivariate_normal((-0.3,-0.3), 0.04*I, 20)
y_1 = -1 * np.ones(20)

# Generate random class '+1'
x_2 = np.random.multivariate_normal((0.15,0.15), 0.01*I, 20)
y_2 = np.ones(20)

# Combine two classes to give the data and label
X_random = np.array(list(x_1) + list(x_2))
Y_random = np.array(list(y_1) + list(y_2))

# Give a sequence of c and find the best cluster of data
c2 = np.linspace(0, 2000, 2001)[1:]
random_c, random_cor_rate, random_y = best_cluster(X_random, Y_random, c2)
print(f'The best c is {random_c} and it gives a correctness of{random_cor_rate: .2%}')

# Plot of the comparision
plt.figure(figsize=(12,8))

# Plot of original data
plt.subplot(211)
plt.scatter(X_random[:, 0], X_random[:, 1], c = Y_random)
plt.title('Original data')

# Plot of spectral clustering
plt.subplot(212)
plt.scatter(X_random[:, 0], X_random[:, 1], c = random_y)
plt.title('Spectral clustering algorithm')

plt.savefig('P2E2')

### Experiment 3

# Load data
dtrain123 = np.loadtxt('C:/Users/james007/Desktop/dtrain123.dat.txt')

# Define dtrain13 to only contain data with label (1,3)
dtrain13 = []
for i in range(dtrain123.shape[0]):
    if dtrain123[i, 0] == 1 or dtrain123[i, 0] == 3:
        dtrain13.append(dtrain123[i])
dtrain13 = np.array(dtrain13)
        
X_dt = dtrain13[:, 1:]
Y_dt = dtrain13[:, 0]

#Relabel the data, change label (1,3) to label (1,-1)
classy = np.ones(Y_dt.shape)
for i in range(len(Y_dt)):
    classy[i] = 2 * (Y_dt[i] == 1) -1 

# Give a sequence of c and find the best cluster of data
# Give the result of best c and best CP
c3 = np.linspace(0, 0.1, 11)
CP = np.ones(len(c3))
for i in range(len(c3)):
    y = spectral_cluster(X_dt, c3[i])
    cor = max((y == classy).sum(), (-y == classy).sum())
    CP[i] = cor / len(classy)
best_CP = np.max(CP)
best_c = c3[np.argmax(CP)]
print(f'The best c is {best_c} and the corresponding CP is{best_CP: .5f}')

# Plot of correctness vs c
plt.figure(figsize=(12,6))
plt.plot(c3, CP)
plt.xlabel('Parameter (scale factor)')
plt.ylabel('Correct cluster percentage') 
plt.title('Model Selection of parameters')
plt.scatter(c3, CP, marker='*')

plt.savefig('P2E3.png')

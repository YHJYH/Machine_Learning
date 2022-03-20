# Import libraries

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from datetime import datetime

# Data Generation function

def sample_01(m,n):
    # Generate samples from 0 and 1
    X = np.random.choice([0,1], (m,n))
    Y = X[:, 0]
    return X, Y

def sample_11(m,n):
    # Generate samples from -1 and 1
    X = np.random.choice([-1,1], (m,n))
    Y = X[:, 0]
    return X, Y

# Four algorithms

def perceptron(m, n):
    # Perceptron algorithm: We generate the data and predict by w@x.
    # Then check if the result is correct and update correspondingly.
    X_train, Y_train = sample_11(m, n)
    X_test, Y_test = sample_11(test_size, n)
    W = np.zeros(n)
    Y_pred = np.zeros(m)
    for t in range(m):
        Y_pred[t] = np.sign(W @ X_train[t])
        if Y_pred[t] * Y_train[t] <= 0:
            W += Y_train[t] * X_train[t]

    # Use the final W to predict and calculate the error.
    Y_res = np.sign(X_test @ W)
    error = np.count_nonzero(Y_res != Y_test)
    return Y_res, error 

def winnow(m, n):
    # Perceptron algorithm: We generate the data and predict by checking whether w@x is less than n.
    # Then check if the result is correct and update correspondingly.
    X_train, Y_train = sample_01(m, n)
    X_test, Y_test = sample_01(test_size, n)
    W = np.ones(n)
    Y_pred = np.ones(m)
    for t in range(m):
        if W @ X_train[t] < n:
            Y_pred[t] = 0
        if Y_pred[t] != Y_train[t]:
            W *= np.float_power(2, (Y_train[t] - Y_pred[t]) * X_train[t])

    # Use the final W to predict and calculate the error.
    Y_res = np.where(X_test @ W < n, 0, 1)
    error = np.count_nonzero(Y_res != Y_test)
    return Y_res, error

def least_squares(m, n):
    # Least squares algorithm: We generate the data and predict by sign(x@w).
    # Then calculate the error.
    X_train, Y_train = sample_11(m, n)
    X_test, Y_test = sample_11(test_size, n)
    W = np.linalg.pinv(X_train) @ Y_train
    Y_res = np.sign(X_test @ W)
    error = np.count_nonzero(Y_res != Y_test)
    return Y_res, error

def one_NN(m, n):
    # One nearest neighbor algorithm: We generate the data and calculate the distance
    # between every pair of train and test point, classify the test point as the nearest train point label.
    X_train, Y_train = sample_11(m, n)
    X_test, Y_test = sample_11(test_size, n)
    Y_res = np.zeros(test_size)
    for i in range(test_size):
        distance = np.linalg.norm(X_train - X_test[i], ord = 2, axis = 1)
        Y_res[i] = Y_train[np.argmin(distance)] 
    error = np.count_nonzero(Y_res != Y_test)
    return Y_res, error

# My method of finding sample complexity.
def complexity(algorithm, max_n):
    ''' My algorithm: try different m for each n no more than max_n for 10 runs
        and calculate the mean error to see whether it is <= 0.1. 
        Use the final m for n as the starting m for n+1.

        Input: 
            algorithm(string): name of the algorithm.
            max_n(int): the maximum of n.
        
        Return:
            res(list): the final m for each n no more than max_n.
    '''
    res = np.zeros(max_n)
    m = 1
    for n in range(max_n):
        while m < 10000:
            error = []
            for i in range(runs):
                error.append(algorithm(m, n+1)[1])
            if np.mean(error) / test_size <= 0.1:
                res[n] = m
                break
            m += 1
    return res

# The normal method of finding sample complexity.
def complex_m(algorithm, max_n):
    ''' Normal algorithm: try different m for each n no more than max_n for 10 runs
        and calculate the mean error to see whether it is <= 0.1. 
        Use 1 as the starting m for each n.

        Input: 
            algorithm(string): name of the algorithm.
            max_n(int): the maximum of n.
        
        Return:
            res(list): the final m for each n no more than max_n.
    '''
    res = np.zeros(max_n)
    for n in range(max_n):
        m = 1
        while m < 10000:
            error = []
            for i in range(runs):
                error.append(algorithm(m, n+1)[1])
            if np.mean(error) / test_size <= 0.1:
                res[n] = m
                break
            m += 1
    return res

# Plot functions

def plot_complex(algorithm, max_n):
    ''' Plot the graph of the called algorithm.

        Input: 
            algorithm(string): name of the algorithm.
            max_n(int): the maximum of n.
    '''
    sequence_n = [i for i in range(1, max_n+1) ]
    res = complexity(algorithm, max_n)
    plt.plot(sequence_n, res)
    plt.xlabel('Dimension (n)')
    plt.ylabel('Estimated number of samples (m)')
    plt.title(f'm to obtain 10% generalisation error versus n for {algorithm.__name__}.')
    plt.show()

def plot_all(algorithm_seq, max_n_seq):
    ''' Plot the graphs of the all algorithms contained in the algorithm_seq.
        And use the corresponding max_n in the max_n_seq.

        Input: 
            algorithm_seq(list): list of names of the algorithms.
            max_n_seq(list): list of all maximum value of n for each algorithm.
    '''
    fig = plt.figure(figsize=(12,8)) 
    gs = gridspec.GridSpec(2, 2)
    ax = [algorithm_seq[k].__name__ for k in range(4)]
    ax[0] = plt.subplot(gs[0,0])
    ax[1] = plt.subplot(gs[0,1])
    ax[2] = plt.subplot(gs[1,0])
    ax[3] = plt.subplot(gs[1,1])
    
    for k in range(4):
        algorithm = algorithm_seq[k]
        res = complexity(algorithm, max_n_seq[k])
        sequence_n = np.array([i for i in range(1, max_n_seq[k]+1)])

        ax[k].plot(sequence_n, res)
        ax[k].set(xlabel = 'n', ylabel = 'm')
        ax[k].set_title(f'm versus n for {algorithm.__name__}')
        
    plt.tight_layout()
    fig.savefig('algorithm15.png')

def plot_fit(algorithm_seq, max_n_seq):
    ''' Plot the graphs of the all algorithms contained in the algorithm_seq.
        And use the corresponding max_n in the max_n_seq.
        Also plot the fitted line for each graph using the specified function form.

        Input: 
            algorithm_seq(list): list of names of the algorithms.
            max_n_seq(list): list of all maximum value of n for each algorithm.
        
        Return:
            Duration(list): the computation time for each algorithm.
    '''
    fig = plt.figure(figsize=(12,8)) 
    gs = gridspec.GridSpec(2, 2)
    ax = [algorithm_seq[k].__name__ for k in range(4)]
    ax[0] = plt.subplot(gs[0,0])
    ax[1] = plt.subplot(gs[0,1])
    ax[2] = plt.subplot(gs[1,0])
    ax[3] = plt.subplot(gs[1,1])
    
    Duration = []
    for k in range(4):
        algorithm = algorithm_seq[k]
        sequence_n = np.array([i for i in range(1, max_n_seq[k]+1)])

        start_time = datetime.now()
        res = complexity(algorithm, max_n_seq[k])
        end_time = datetime.now()
        Duration.append(end_time - start_time) 
        
        if k == 1:
            a, b = np.polyfit(np.log(sequence_n), res, 1)
            y = a * np.log(sequence_n) + b
        elif k == 3:
            a, b = np.polyfit(sequence_n, np.log(res), 1)
            y = np.exp(a * sequence_n + b)
        else:
            a, b = np.polyfit(sequence_n, res, 1)
            y = a * sequence_n + b
    
        ax[k].plot(sequence_n, y, label = f'a = {a: 2f}, b = {b: 2f}')
        ax[k].plot(sequence_n, res)
        ax[k].set(xlabel = 'n', ylabel = 'm')
        ax[k].set_title(f'm versus n for {algorithm.__name__}')
        ax[k].legend()
        
    plt.tight_layout()
    fig.savefig('algorithm15 fit.png')
    return Duration

def plot_comparison(algorithm_seq, max_n_seq):
    ''' Plot the graphs of the all algorithms contained in the algorithm_seq.
        And use the corresponding max_n in the max_n_seq.
        Use both my method and the normal method to produce m and see the comparison in graphs.

        Input: 
            algorithm_seq(list): list of names of the algorithms.
            max_n_seq(list): list of all maximum value of n for each algorithm.
        
        Return:
            Duration1(list): the computation time for each algorithm using my method.
            Duration2(list): the computation time for each algorithm using the normal method.
    '''
    fig = plt.figure(figsize=(12,8)) 
    gs = gridspec.GridSpec(2, 2)
    ax = [algorithm_seq[k].__name__ for k in range(4)]
    ax[0] = plt.subplot(gs[0,0])
    ax[1] = plt.subplot(gs[0,1])
    ax[2] = plt.subplot(gs[1,0])
    ax[3] = plt.subplot(gs[1,1])
    
    Duration1 = []
    Duration2 = []
    for k in range(4):
        algorithm = algorithm_seq[k]
        sequence_n = np.array([i for i in range(1, max_n_seq[k]+1)])
        
        start_time = datetime.now()
        res1 = complexity(algorithm, max_n_seq[k])
        end_time = datetime.now()
        Duration1.append(end_time - start_time) 

        start_time = datetime.now()
        res2 = complex_m(algorithm, max_n_seq[k])
        end_time = datetime.now()
        Duration2.append(end_time - start_time) 
       

        ax[k].plot(sequence_n, res1, label = 'generate m from the m for n-1')
        ax[k].plot(sequence_n, res2, label = 'generate m from 1')
        ax[k].set(xlabel = 'n', ylabel = 'm')
        ax[k].set_title(f'm versus n for {algorithm.__name__}')
        ax[k].legend()
        
    plt.tight_layout()
    fig.savefig('algorithm15 comparison.png')
    return Duration1, Duration2

# Specify the parameters not included in the algorithm.
runs = 10
test_size = 5000
max_n_seq = [100, 100, 100, 15]
algorithm_seq = [perceptron, winnow, least_squares, one_NN]

# Run the algorithms and plot
plot_all(algorithm_seq, max_n_seq)
plot_fit(algorithm_seq, max_n_seq)
plot_comparison(algorithm_seq, max_n_seq) 

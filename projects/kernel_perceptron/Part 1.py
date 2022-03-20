# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 03:45:39 2021

@author: 21049846

0078 SL CW2
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
#from collections import Counter
#from sklearn.preprocessing import OneHotEncoder
import math
import statistics as stat
import time
from scipy.spatial.distance import cdist
from itertools import combinations
#from numba import jit, cuda

dataset = np.loadtxt('zipcombo.dat')
t = dataset[:,0].reshape((-1,1))
X = dataset[:,1:]


class_dict = {}
for i in range(int(min(t)),int(max(t)+1)):
    class_dict[str(i)] = np.count_nonzero(t==i)
num_classes = len(class_dict)


def shuffle(train, label, rs):
    """
    split zipcombo.dat into training and test set.
    """
    full_data_list = []
    full_data_list.append(train_test_split(train, label, test_size=0.2, random_state=rs))
    return full_data_list[0]
    
def polynomial_gram_matrix(X1, X2, d):
    """
    :input:
        X1: input matrix.
        X2: input matrix.
        d: degree of polynomial kernel function.
    :output:
        K: gram matrix
    """
    K = np.matmul(X1,X2.T)**d
    return K
    
def gaussian_gram_matrix(X1, X2, c):
    
    K = np.exp(-c * cdist(X1,X2)**2) # problem
    
    return K
    
def test_kernel_perceptron(train_set, label, d):
    run = 20
    epoches = 20
    k = 10  
    train_error = []
    confusion = []
    test_error = []
    confusion_test = []
    
    for r in range(run):
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((k,m))
        K = polynomial_gram_matrix(data[0],data[0],d)
        confusion_matrix = np.zeros((k,k))
        for e in range(epoches):
            num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                pred_label = np.argmax(pred_y_value)
                #print("test")
                if pred_label != true_label:
                    confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                    alpha[int(pred_label),t] -= 1
                    alpha[int(true_label),t] += 1
                    num_false += 1
                #print("current t: {}".format(t))
            train_error_rate = num_false / m
            
            #train_accuracy = 1 - train_error_rate     
        train_error.append(train_error_rate)
        
        confusion.append(confusion_matrix)
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, current train error rate: {}, time taken: {}".format(r+1, train_error[-1], time_taken))
        #print("confidence vector is {}".format(confidence))
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = polynomial_gram_matrix(data[0],data[1],d)
        confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            pred_label = np.argmax(confidence_test)
            if pred_label != true_label:
                num_false_test += 1
                confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error.append(test_error_rate)
        confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error[-1], time_taken_test))
    
    mean_train_error = stat.mean(train_error)
    std_train_error = stat.stdev(train_error)
    avg_confusion_mat = confusion[-1] / sum(confusion)
    
    mean_test_error = stat.mean(test_error)
    std_test_error = stat.stdev(test_error)
    avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
    
    
    
    return [mean_train_error,std_train_error],[mean_test_error,std_test_error], [avg_confusion_mat,avg_confusion_matrix_test]




def k_fold_cross_val(k,data_set):
    """
    :input:
        k: number of folds
        data_set: dataset to be split into k folds
    :output:
        partitioned_set: a list contains k sub-lists, each sub-list is one fold of the dataset
    """
    if len(data_set) % k == 0:
        partitioned_set = np.reshape(data_set,(k,int(len(data_set)/k),-1))
    else:
        new_data_set = data_set.tolist()
        new_length = len(data_set) - len(data_set) % k
        left = [new_data_set[i]  for i in range(new_length, len(data_set))]
        #partitioned_set = [element for element in new_data_set if element not in left]
        partitioned_set = [new_data_set[i] for i in range(new_length)]
        partitioned_set = np.reshape(partitioned_set, (k, int(new_length / k), -1))
        partitioned_set = partitioned_set.tolist()
        for i in range(len(left)):
            partitioned_set[i].append(left[i])
        
    return partitioned_set


def cv_kernel_perceptron(train_set, label):
    run = 20
    epoches = 20
    k = 10  
    #train_error = []
    #confusion = []
    
    best_d = []
    test_error_best_d = []
    
    for r in range(run):
        
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        cv_data_train = k_fold_cross_val(5,data[0])
        cv_data_label = k_fold_cross_val(5,data[2])
        #fold_list = [0,1,2,3,4]
        
        test_error_cv = []
        confusion_test_cv = []
        for d in range(1,8):
            test_error = []
            confusion_test = []
            for i in range(5):
                cv_test = np.array(cv_data_train[i])
                cv_test_label = np.array(cv_data_label[i])
                cv_train = []
                cv_train_label = []
                for j in range(5):
                    if j != i:
                        cv_train = cv_train + cv_data_train[j]
                        cv_train_label = cv_train_label + cv_data_label[j]
                cv_train = np.array(cv_train)
                cv_train_label = np.array(cv_train_label)
                
                cv_m = cv_train.shape[0]
                cv_alpha = np.zeros((k,cv_m))
                cv_K = polynomial_gram_matrix(cv_train,cv_train,d)
                #cv_confusion_matrix = np.zeros((k,k))
                
                for e in range(epoches):
                    #num_false = 0
                    for t in range(cv_m):
                        true_label = cv_train_label[t]
                        pred_y_value = np.dot(cv_alpha,cv_K[:,t].reshape((cv_m,-1)))
                        confidence = pred_y_value
                        pred_label = np.argmax(confidence)
                        #print("test")
                        if pred_label != true_label:
                            #cv_confusion_matrix[int(cv_train_label[t]),int(pred_label)] += 1
                            cv_alpha[int(pred_label),t] -= 1
                            cv_alpha[int(true_label),t] += 1
                            #num_false += 1
                        #print("current t: {}".format(t))
                    #train_error_rate = num_false / cv_m
                    
                    #train_accuracy = 1 - train_error_rate     
                #train_error.append(train_error_rate)
                
                #confusion.append(cv_confusion_matrix)
                end = time.time()
                
                time_taken = end - start
                
                print("current run: {}, time taken: {}".format(r+1, time_taken))
                #print("confidence vector is {}".format(confidence))
                
                start_test = time.time()
                m_test_cv = cv_test.shape[0]
                K_test_cv = polynomial_gram_matrix(cv_train,cv_test,d)
                confusion_matrix_test_cv = np.zeros((k,k))
                num_false_test = 0
                
                for t in range(m_test_cv):
                    true_label = cv_test_label[t]
                    pred_y_value_test = np.dot(cv_alpha, K_test_cv[:,t].reshape((cv_m,-1)))
                    confidence_test = pred_y_value_test
                    pred_label = np.argmax(confidence_test)
                    if pred_label != true_label:
                        num_false_test += 1
                        confusion_matrix_test_cv[int(true_label), int(pred_label)] += 1
                test_error_rate = num_false_test / m_test_cv
                
                test_error.append(test_error_rate)
                confusion_test.append(confusion_matrix_test_cv)
                end_test = time.time()
                
                time_taken_test = end_test - start_test
                print("current run: {}, d: {}, current test error rate: {}, time taken: {}".format(r+1, d, test_error[-1], time_taken_test))
            
            #mean_train_error = stat.mean(train_error)
            #std_train_error = stat.stdev(train_error)
            #avg_confusion_mat = confusion[-1] / sum(confusion)
        
            mean_test_error_cv = stat.mean(test_error)
            test_error_cv.append(mean_test_error_cv)
            #std_test_error = stat.stdev(test_error)
            avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
            confusion_test_cv.append(avg_confusion_matrix_test)
            
        d_star = np.argmin(test_error_cv) + 1
        best_d.append(d_star)
        
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((k,m))
        K = polynomial_gram_matrix(data[0],data[0],d_star)
        #confusion_matrix = np.zeros((k,k))
        for e in range(epoches):
            #num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                pred_label = np.argmax(pred_y_value)
                #print("test")
                if pred_label != true_label:
                    #confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                    alpha[int(pred_label),t] -= 1
                    alpha[int(true_label),t] += 1
                    #num_false += 1
                #print("current t: {}".format(t))
            #train_error_rate = num_false / m
            
            #train_accuracy = 1 - train_error_rate     
        #train_error.append(train_error_rate)
        
        #confusion.append(confusion_matrix)
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, best d: {}, time taken: {}".format(r+1, d_star, time_taken))
        #print("confidence vector is {}".format(confidence))
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = polynomial_gram_matrix(data[0],data[1],d_star)
        confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            pred_label = np.argmax(confidence_test)
            if pred_label != true_label:
                num_false_test += 1
                confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error_best_d.append(test_error_rate)
        confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error_best_d[-1], time_taken_test))
        

    
    return best_d, test_error_best_d

def confusion_kernel_perceptron(train_set, label, d,r):
    #run = 20
    epoches = 20
    k = 10  
    train_error = []
    #confusion = []
    test_error = []
    confusion_test = []
    
    #for r in range(run):
    start = time.time()
        
    data = shuffle(train_set,label,r)
        
    m = data[0].shape[0]
    alpha = np.zeros((k,m))
    K = polynomial_gram_matrix(data[0],data[0],d)
    #confusion_matrix = np.zeros((k,k))
    for e in range(epoches):
        num_false = 0
        for t in range(m):
            true_label = data[2][t]
            pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
            confidence = pred_y_value
            pred_label = np.argmax(confidence)
            #print("test")
            if pred_label != true_label:
                #confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                alpha[int(pred_label),t] -= 1
                alpha[int(true_label),t] += 1
                num_false += 1
            #print("current t: {}".format(t))
        train_error_rate = num_false / m
        
        #train_accuracy = 1 - train_error_rate     
    train_error.append(train_error_rate)
    
    #confusion.append(confusion_matrix)
    end = time.time()
    
    time_taken = end - start
    
    print("train error rate: {}, time taken: {}".format(train_error[-1], time_taken))
    #print("confidence vector is {}".format(confidence))
        
    start_test = time.time()
    m_test = data[1].shape[0]
    K_test = polynomial_gram_matrix(data[0],data[1],d)
    confusion_matrix_test = np.zeros((k,k))
    num_false_test = 0
    
    for t in range(m_test):
        true_label = data[3][t]
        pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
        confidence_test = pred_y_value_test
        pred_label = np.argmax(confidence_test)
        if pred_label != true_label:
            num_false_test += 1
            confusion_matrix_test[int(true_label), int(pred_label)] += 1
    test_error_rate = num_false_test / m_test
    
    test_error.append(test_error_rate)
    #confusion_test.append(confusion_matrix_test)
    end_test = time.time()
    
    time_taken_test = end_test - start_test
    print("current test error rate: {}, time taken: {}".format(test_error[-1], time_taken_test))

    #mean_train_error = stat.mean(train_error)
    #std_train_error = stat.stdev(train_error)
    #avg_confusion_mat = confusion[-1] / sum(confusion)
    
    #mean_test_error = stat.mean(test_error)
    #std_test_error = stat.stdev(test_error)
    #avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
    """
    mean_cm = sum(confusion_test) / 20
    #std_cm = np.zeros((k,k))
    std_cm_list = []
    for i in range(k):
        for j in range(k):
            element_list = []
            for r in range(run):
                element_list.append(confusion_test[r][i,j])
            std = stat.stdev(element_list)
            std_cm_list.append(std)
    std_cm = np.array(std_cm_list).reshape((k,k))
    """
    
    return confusion_matrix_test

def test_gaussian_perceptron(train_set, label, c):
    run = 20
    epoches = 20
    k = 10  
    train_error = []
    confusion = []
    test_error = []
    confusion_test = []
    
    for r in range(run):
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((k,m))
        K = gaussian_gram_matrix(data[0],data[0],c)
        confusion_matrix = np.zeros((k,k))
        for e in range(epoches):
            num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                pred_label = np.argmax(confidence)
                #print("test")
                if pred_label != true_label:
                    confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                    alpha[int(pred_label),t] -= 1
                    alpha[int(true_label),t] += 1
                    num_false += 1
                #print("current t: {}".format(t))
            train_error_rate = num_false / m
            
            #train_accuracy = 1 - train_error_rate     
        train_error.append(train_error_rate)
        
        confusion.append(confusion_matrix)
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, current train error rate: {}, time taken: {}".format(r+1, train_error[-1], time_taken))
        #print("confidence vector is {}".format(confidence))
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = gaussian_gram_matrix(data[0],data[1],c)
        confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            pred_label = np.argmax(confidence_test)
            if pred_label != true_label:
                num_false_test += 1
                confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error.append(test_error_rate)
        confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error[-1], time_taken_test))
    
    mean_train_error = stat.mean(train_error)
    std_train_error = stat.stdev(train_error)
    #avg_confusion_mat = confusion[-1] / sum(confusion)
    
    mean_test_error = stat.mean(test_error)
    std_test_error = stat.stdev(test_error)
    #avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
    
    
    
    return [mean_train_error,std_train_error],[mean_test_error,std_test_error]

def cv_gaussian_perceptron(train_set, label):
    run = 20
    epoches = 20
    k = 10  
    #train_error = []
    #confusion = []
    
    best_c = []
    test_error_best_c = []
    c = [0.005,0.010,0.015,0.020,0.025,0.030,0.035]
    
    for r in range(run):
        
        #start = time.time()
        
        data = shuffle(train_set,label,r)
        
        cv_data_train = k_fold_cross_val(5,data[0])
        cv_data_label = k_fold_cross_val(5,data[2])
        #fold_list = [0,1,2,3,4]
        
        test_error_cv = []
        confusion_test_cv = []
        for d in range(0,7):
            test_error = []
            confusion_test = []
            for i in range(5):
                cv_test = np.array(cv_data_train[i])
                cv_test_label = np.array(cv_data_label[i])
                cv_train = []
                cv_train_label = []
                for j in range(5):
                    if j != i:
                        cv_train = cv_train + cv_data_train[j]
                        cv_train_label = cv_train_label + cv_data_label[j]
                cv_train = np.array(cv_train)
                cv_train_label = np.array(cv_train_label)
                
                cv_m = cv_train.shape[0]
                cv_alpha = np.zeros((k,cv_m))
                cv_K = gaussian_gram_matrix(cv_train,cv_train,c[d])
                #cv_confusion_matrix = np.zeros((k,k))
                
                start = time.time()
                
                for e in range(epoches):
                    #num_false = 0
                    for t in range(cv_m):
                        true_label = cv_train_label[t]
                        pred_y_value = np.dot(cv_alpha,cv_K[:,t].reshape((cv_m,-1)))
                        confidence = pred_y_value
                        pred_label = np.argmax(confidence)
                        #print("test")
                        if pred_label != true_label:
                            #cv_confusion_matrix[int(cv_train_label[t]),int(pred_label)] += 1
                            cv_alpha[int(pred_label),t] -= 1
                            cv_alpha[int(true_label),t] += 1
                            #num_false += 1
                        #print("current t: {}".format(t))
                    #train_error_rate = num_false / cv_m
                    
                    #train_accuracy = 1 - train_error_rate     
                #train_error.append(train_error_rate)
                
                #confusion.append(cv_confusion_matrix)
                end = time.time()
                
                time_taken = end - start
                
                print("current run: {}, time taken: {}".format(r+1, time_taken))
                #print("confidence vector is {}".format(confidence))
                
                start_test = time.time()
                m_test_cv = cv_test.shape[0]
                K_test_cv = gaussian_gram_matrix(cv_train,cv_test,c[d])
                confusion_matrix_test_cv = np.zeros((k,k))
                num_false_test = 0
                
                for t in range(m_test_cv):
                    true_label = cv_test_label[t]
                    pred_y_value_test = np.dot(cv_alpha, K_test_cv[:,t].reshape((cv_m,-1)))
                    confidence_test = pred_y_value_test
                    pred_label = np.argmax(confidence_test)
                    if pred_label != true_label:
                        num_false_test += 1
                        confusion_matrix_test_cv[int(true_label), int(pred_label)] += 1
                test_error_rate = num_false_test / m_test_cv
                
                test_error.append(test_error_rate)
                confusion_test.append(confusion_matrix_test_cv)
                end_test = time.time()
                
                time_taken_test = end_test - start_test
                print("current run: {}, c: {}, current test error rate: {}, time taken: {}".format(r+1, c[d], test_error[-1], time_taken_test))
            
            #mean_train_error = stat.mean(train_error)
            #std_train_error = stat.stdev(train_error)
            #avg_confusion_mat = confusion[-1] / sum(confusion)
        
            mean_test_error_cv = stat.mean(test_error)
            test_error_cv.append(mean_test_error_cv)
            #std_test_error = stat.stdev(test_error)
            avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
            confusion_test_cv.append(avg_confusion_matrix_test)
            
        d_star = np.argmin(test_error_cv) + 1
        c_star = c[d_star]
        best_c.append(c_star)
        
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((k,m))
        K = gaussian_gram_matrix(data[0],data[0],c_star)
        #confusion_matrix = np.zeros((k,k))
        for e in range(epoches):
            #num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                pred_label = np.argmax(pred_y_value)
                #print("test")
                if pred_label != true_label:
                    #confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                    alpha[int(pred_label),t] -= 1
                    alpha[int(true_label),t] += 1
                    #num_false += 1
                #print("current t: {}".format(t))
            #train_error_rate = num_false / m
            
            #train_accuracy = 1 - train_error_rate     
        #train_error.append(train_error_rate)
        
        #confusion.append(confusion_matrix)
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, best c: {}, time taken: {}".format(r+1, c_star, time_taken))
        #print("confidence vector is {}".format(confidence))
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = gaussian_gram_matrix(data[0],data[1],c_star)
        confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            pred_label = np.argmax(confidence_test)
            if pred_label != true_label:
                num_false_test += 1
                confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error_best_c.append(test_error_rate)
        confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error_best_c[-1], time_taken_test))
        

    
    return best_c, test_error_best_c

def hardest_to_predict(train_set, label, d,r):
    #run = 20
    epoches = 20
    k = 10  
    #train_error = []
    #confusion = []
    #test_error = []
    #confusion_test = []
    
    
    #for r in range(run):
    start = time.time()
        
    data = shuffle(train_set,label,r)
        
    m = data[0].shape[0]
    alpha = np.zeros((k,m))
    K = polynomial_gram_matrix(data[0],data[0],d)
    #confusion_matrix = np.zeros((k,k))
    for e in range(epoches):
        #num_false = 0
        for t in range(m):
            true_label = data[2][t]
            pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
            confidence = pred_y_value
            pred_label = np.argmax(confidence)
            #print("test")
            if pred_label != true_label:
                #confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                alpha[int(pred_label),t] -= 1
                alpha[int(true_label),t] += 1
                #num_false += 1
            #print("current t: {}".format(t))
        #train_error_rate = num_false / m
        
        #train_accuracy = 1 - train_error_rate     
    #train_error.append(train_error_rate)
    
    #confusion.append(confusion_matrix)
    end = time.time()
    
    time_taken = end - start
        
    print("current r: {}, d: {}, time taken: {}".format(r+1, d, time_taken))
    #print("confidence vector is {}".format(confidence))
        
    start_test = time.time()
    m_whole = train_set.shape[0]
    misprediction = np.zeros((m_whole,1))
    #m_test = data[1].shape[0]
    K_test = polynomial_gram_matrix(data[0],train_set,d)
    #confusion_matrix_test = np.zeros((k,k))
    #num_false_test = 0
        
    for t in range(m_whole):
        true_label = label[t]
        pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
        confidence_test = pred_y_value_test
        pred_label = np.argmax(confidence_test)
        if pred_label != true_label:
            misprediction[t] += 1
            #num_false_test += 1
            #confusion_matrix_test[int(true_label), int(pred_label)] += 1
    #test_error_rate = num_false_test / m_test
    
    #test_error.append(test_error_rate)
    #confusion_test.append(confusion_matrix_test)
    end_test = time.time()
    
    time_taken_test = end_test - start_test
    print("current r: {}, time taken: {}".format(r+1, time_taken_test))
    
    #mean_train_error = stat.mean(train_error)
    #std_train_error = stat.stdev(train_error)
    #avg_confusion_mat = confusion[-1] / sum(confusion)
    
    #mean_test_error = stat.mean(test_error)
    #std_test_error = stat.stdev(test_error)
    # avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
    
    
    
    return misprediction


def OvO_kernel_test(train_set, label, d):
    
    run = 20
    epoches = 20
    k = 10  
    classifier = list(combinations(range(k),2))
    alpha_converter = dict(zip(tuple(classifier), tuple(range(len(classifier)))))
    
    train_error = []    
    test_error = []
    
    
    for r in range(run):
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((int(k*(k-1)/2),m))
        K = polynomial_gram_matrix(data[0],data[0],d)
        
        for e in range(epoches):
            num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                binary_classifier = np.sign(confidence).clip(0,None)
                
                classifiers = np.zeros((len(classifier)))
                for c_index, b_index in enumerate(binary_classifier):
                    classifiers[c_index] = classifier[int(c_index)][int(b_index)]
                
                pred_label = np.bincount(classifiers.astype(int)).argmax()
                if pred_label != true_label:
                    num_false += 1
                    
                for loc, (a,b) in enumerate(alpha_converter.keys()):
                    pred_y = classifiers[loc]
                        
                    if true_label == a and pred_y != true_label:
                        alpha[loc,t] -= 1
                    if true_label == b and pred_y != true_label:
                        alpha[loc,t] += 1
                
            train_error_rate = num_false / m
             
        train_error.append(train_error_rate)
        
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, current train error rate: {}, time taken: {}".format(r+1, train_error[-1], time_taken))
        
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = polynomial_gram_matrix(data[0],data[1],d)
        #confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            
            binary_classifier_test = np.sign(confidence_test).clip(0,None)
            classifiers_test = np.zeros((len(classifier)))
            for c_index, b_index in enumerate(binary_classifier_test):
                classifiers_test[c_index] = classifier[int(c_index)][int(b_index)]
            
            pred_label_test = np.bincount(classifiers_test.astype(int)).argmax()
            
            if pred_label_test != true_label:
                num_false_test += 1
                #confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error.append(test_error_rate)
        #confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error[-1], time_taken_test))
    
    mean_train_error = stat.mean(train_error)
    std_train_error = stat.stdev(train_error)
    #avg_confusion_mat = confusion[-1] / sum(confusion)
    
    mean_test_error = stat.mean(test_error)
    std_test_error = stat.stdev(test_error)
    #avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
    
    
    return [mean_train_error,std_train_error],[mean_test_error,std_test_error]

def cv_OvO_perceptron(train_set, label):
    run = 20
    epoches = 20
    k = 10  
    
    classifier = list(combinations(range(k),2))
    alpha_converter = dict(zip(tuple(classifier), tuple(range(len(classifier)))))
    
    best_d = []
    test_error_best_d = []
    
    for r in range(run):
        
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        cv_data_train = k_fold_cross_val(5,data[0])
        cv_data_label = k_fold_cross_val(5,data[2])
        
        test_error_cv = []
        #confusion_test_cv = []
        for d in range(1,8):
            test_error = []
            #confusion_test = []
            for i in range(5):
                cv_test = np.array(cv_data_train[i])
                cv_test_label = np.array(cv_data_label[i])
                cv_train = []
                cv_train_label = []
                for j in range(5):
                    if j != i:
                        cv_train = cv_train + cv_data_train[j]
                        cv_train_label = cv_train_label + cv_data_label[j]
                cv_train = np.array(cv_train)
                cv_train_label = np.array(cv_train_label)
                
                cv_m = cv_train.shape[0]
                cv_alpha = np.zeros((int(k*(k-1)/2),cv_m))
                cv_K = polynomial_gram_matrix(cv_train,cv_train,d)
                
                for e in range(epoches):
                    #num_false = 0
                    for t in range(cv_m):
                        true_label = cv_train_label[t]
                        pred_y_value = np.dot(cv_alpha,cv_K[:,t].reshape((cv_m,-1)))
                        confidence = pred_y_value
                        binary_classifier = np.sign(confidence).clip(0,None)
                        
                        classifiers = np.zeros((len(classifier)))
                        for c_index, b_index in enumerate(binary_classifier):
                            classifiers[c_index] = classifier[int(c_index)][int(b_index)]
                        
                        pred_label = np.bincount(classifiers.astype(int)).argmax()
                        #pred_label = np.argmax(confidence)
                        #print("test")
                        #if pred_label != true_label:
                            #cv_confusion_matrix[int(cv_train_label[t]),int(pred_label)] += 1
                            #num_false += 1
                        for loc, (a,b) in enumerate(alpha_converter.keys()):
                            pred_y = classifiers[loc]
                                
                            if true_label == a and pred_y != true_label:
                                cv_alpha[loc,t] -= 1
                            if true_label == b and pred_y != true_label:
                                cv_alpha[loc,t] += 1
                        #print("current t: {}".format(t))
                    #train_error_rate = num_false / cv_m
                    
                    #train_accuracy = 1 - train_error_rate     
                #train_error.append(train_error_rate)
                
                #confusion.append(cv_confusion_matrix)
                end = time.time()
                
                time_taken = end - start
                
                print("current run: {}, time taken: {}".format(r+1, time_taken))
                #print("confidence vector is {}".format(confidence))
                
                start_test = time.time()
                m_test_cv = cv_test.shape[0]
                K_test_cv = polynomial_gram_matrix(cv_train,cv_test,d)
                #confusion_matrix_test_cv = np.zeros((k,k))
                num_false_test = 0
                
                for t in range(m_test_cv):
                    true_label = cv_test_label[t]
                    pred_y_value_test = np.dot(cv_alpha, K_test_cv[:,t].reshape((cv_m,-1)))
                    confidence_test = pred_y_value_test
                    binary_classifier_test = np.sign(confidence_test).clip(0,None)
                    classifiers_test = np.zeros((len(classifier)))
                    for c_index, b_index in enumerate(binary_classifier_test):
                        classifiers_test[c_index] = classifier[int(c_index)][int(b_index)]
                    
                    pred_label_test = np.bincount(classifiers_test.astype(int)).argmax()
                    
                    #pred_label = np.argmax(confidence_test)
                    if pred_label_test != true_label:
                        num_false_test += 1
                        #confusion_matrix_test_cv[int(true_label), int(pred_label)] += 1
                test_error_rate = num_false_test / m_test_cv
                
                test_error.append(test_error_rate)
                #confusion_test.append(confusion_matrix_test_cv)
                end_test = time.time()
                
                time_taken_test = end_test - start_test
                print("current run: {}, d: {}, current test error rate: {}, time taken: {}".format(r+1, d, test_error[-1], time_taken_test))
            
            #mean_train_error = stat.mean(train_error)
            #std_train_error = stat.stdev(train_error)
            #avg_confusion_mat = confusion[-1] / sum(confusion)
        
            mean_test_error_cv = stat.mean(test_error)
            test_error_cv.append(mean_test_error_cv)
            #std_test_error = stat.stdev(test_error)
            #avg_confusion_matrix_test = confusion_test[-1] / sum(confusion_test)
            #confusion_test_cv.append(avg_confusion_matrix_test)
            
        d_star = np.argmin(test_error_cv) + 1
        best_d.append(d_star)
        
        start = time.time()
        
        data = shuffle(train_set,label,r)
        
        m = data[0].shape[0]
        alpha = np.zeros((int(k*(k-1)/2),m))
        K = polynomial_gram_matrix(data[0],data[0],d_star)
        #confusion_matrix = np.zeros((k,k))
        for e in range(epoches):
            num_false = 0
            for t in range(m):
                true_label = data[2][t]
                pred_y_value = np.dot(alpha,K[:,t].reshape((m,-1)))
                confidence = pred_y_value
                binary_classifier = np.sign(confidence).clip(0,None)
                
                classifiers = np.zeros((len(classifier)))
                for c_index, b_index in enumerate(binary_classifier):
                    classifiers[c_index] = classifier[int(c_index)][int(b_index)]
                pred_label = np.bincount(classifiers.astype(int)).argmax()
                #pred_label = np.argmax(pred_y_value)
                #print("test")
                if pred_label != true_label:
                    
                    #confusion_matrix[int(data[2][t]),int(pred_label)] += 1
                    #alpha[int(pred_label),t] -= 1
                    #alpha[int(true_label),t] += 1
                    num_false += 1
                for loc, (a,b) in enumerate(alpha_converter.keys()):
                    pred_y = classifiers[loc]
                        
                    if true_label == a and pred_y != true_label:
                        alpha[loc,t] -= 1
                    if true_label == b and pred_y != true_label:
                        alpha[loc,t] += 1
                #print("current t: {}".format(t))
            #train_error_rate = num_false / m
            
            #train_accuracy = 1 - train_error_rate     
        #train_error.append(train_error_rate)
        
        #confusion.append(confusion_matrix)
        end = time.time()
        
        time_taken = end - start
        
        print("current run: {}, best d: {}, time taken: {}".format(r+1, d_star, time_taken))
        #print("confidence vector is {}".format(confidence))
        
        start_test = time.time()
        m_test = data[1].shape[0]
        K_test = polynomial_gram_matrix(data[0],data[1],d_star)
        #confusion_matrix_test = np.zeros((k,k))
        num_false_test = 0
        
        for t in range(m_test):
            true_label = data[3][t]
            pred_y_value_test = np.dot(alpha, K_test[:,t].reshape((m,-1)))
            confidence_test = pred_y_value_test
            binary_classifier_test = np.sign(confidence_test).clip(0,None)
            classifiers_test = np.zeros((len(classifier)))
            for c_index, b_index in enumerate(binary_classifier_test):
                classifiers_test[c_index] = classifier[int(c_index)][int(b_index)]
            
            pred_label_test = np.bincount(classifiers_test.astype(int)).argmax()
            #pred_label = np.argmax(confidence_test)
            if pred_label_test != true_label:
                num_false_test += 1
                #confusion_matrix_test[int(true_label), int(pred_label)] += 1
        test_error_rate = num_false_test / m_test
        
        test_error_best_d.append(test_error_rate)
        #confusion_test.append(confusion_matrix_test)
        end_test = time.time()
        
        time_taken_test = end_test - start_test
        print("current run: {}, current test error rate: {}, time taken: {}".format(r+1, test_error_best_d[-1], time_taken_test))
        

    
    return best_d, test_error_best_d


if __name__ == "__main__":
    #xx = np.loadtxt('dtrain123.dat')
    #xx_train = xx[:,1:]
    #xx_label = xx[:,0].reshape((-1,1))
    #onehot_encoder_x = OneHotEncoder(sparse=False)
    #xx_encoded = onehot_encoder_x.fit_transform(xx_label)
    #for i in range(len(xx_label)):
        #xx_encoded[i] = np.where(xx_encoded[i] == 0, -1, xx_encoded[i])
    print("start")
    """
    print(class_dict)
    
    fig, axes = plt.subplots(2,5)
    axes = axes.flatten()
    for i in range(num_classes):
        index = np.where(t == i)
        image = np.reshape(X[index], (16, 16))
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()
    
    for i in range(1,8):
        print("test_kernel_perceptron(X,t,{})".format(i))
        test_kernel_perceptron(X,t,i)
    
    c = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]
    for j in range(8):
        print("test_gaussian_perceptron(X,t,{})".format(c[j]))
        test_gaussian_perceptron(X,t,c[j])
    
    
    print("cv_kernel_perceptron(X,t)")
    best_d, test_error_d = cv_kernel_perceptron(X,t)
    best_c, test_error_c = cv_gaussian_perceptron(X,t)
    
    mean_d, std_d = stat.mean(best_d), stat.stdev(best_d)
    mean_error_d, std_error_d = stat.mean(test_error_d), stat.stdev(test_error_d)
    mean_c, std_c = stat.mean(best_c), stat.stdev(best_c)
    mean_error_c, std_error_c = stat.mean(test_error_c), stat.stdev(test_error_c)
    
    
    # best_d = [5,5,4,5,3,4,4,4,4,7,5,3,5,4,4,3,4,7,4,5]
    # best_c = [0.02,0.02, 0.025,0.02,0.02,0.025,0.015,0.015,0.02,0.02,0.03,0.02,0.02,0.015,0.02,0.025,0.02, 0.025,0.015,0.02]

    confusion_matrix_list = []
    misclassified = []
    for i in range(len(best_d)):
        misclassified.append(hardest_to_predict(X,t,best_d[i],i))
        cmt = confusion_kernel_perceptron(X,t,best_d[i],i)
        norm_cmt = cmt / np.sum(cmt)
        confusion_matrix_list.append(norm_cmt)
        
       
    mean_cm = np.array(sum(confusion_matrix_list)) / len(confusion_matrix_list)
    std_cm_list = []
    for i in range(10):
        for j in range(10):
            element_list = []
            for r in range(len(best_d)):
                element_list.append(confusion_matrix_list[r][i,j])
            std = stat.stdev(element_list)
            std_cm_list.append(std)
    std_cm = np.array(std_cm_list).reshape((10,10))
    
    HtP_index = np.argpartition(sum(misclassified).reshape((len(sum(misclassified)),)),-5)[-5:]
    
    for i in range(len(HtP_index)):
        index = int(HtP_index[i])
        
        pics = np.reshape(X[index],(16,16))
        
        plt.imshow(pics, cmap='gray')
        plt.axis('off')
        plt.title("Label: {}, misclassified {} times".format(t[index],sum(misclassified)[index]))
        plt.show()
        
    for i in range(1,8):
        print("OvO_kernel_test(X,t,{})".format(i))
        OvO_kernel_test(X,y,i)
    
    ovo_d. ovo_test_error = cv_OvO_perceptron(X,t) 
    mean_ovo_d, std_ovo_d = stat.mean(ovo_d), stat.stdev(ovo_d)
    mean_ovo_error, std_ovo_error = stat.mean(ovo_test_error), stat.stdev(ovo_test_error)
    """
    
    
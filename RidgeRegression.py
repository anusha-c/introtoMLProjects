#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:50:41 2020

@author: zhe
"""

# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    dataset = np.loadtxt(filename)
    y = dataset[:,-1]
    x = dataset[:, 0:-1]
    return x, y


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    # your code
    x_lim = int(len(x)*train_proportion)
    y_lim = int(len(y)*train_proportion)
    x_train = x[0:x_lim]
    y_train = y[0:y_lim]
    x_test = x[x_lim:-1]
    y_test = y[y_lim:-1]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation, check our lecture slides
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    # your code
    beta = np.linalg.inv(x.T.dot(x) + lambdaV*(np.identity(x.shape[1]))).dot(x.T.dot(y))
    return beta


# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    # your code
    y_diff_squared = (y-y_predict)**2
    loss = np.mean(y_diff_squared)
    return loss

# Given an array of x and theta predict y
def predict(x, theta):
    # your code
    y_predict = np.matmul(x, theta)
    return y_predict

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
    valid_losses = np.zeros(len(lambdas))
    training_losses = np.zeros(len(lambdas))
    data_split = []
    x_half1, x_half2, y_half1, y_half2 = train_test_split(x_train, y_train, 0.5)
    x_quarter1, x_quarter2, y_quarter1, y_quarter2 = train_test_split(x_half1, y_half1, 0.5)
    x_quarter3, x_quarter4, y_quarter3, y_quarter4 = train_test_split(x_half2, y_half2, 0.5)
    data_split.append([x_quarter1, y_quarter1])
    data_split.append([x_quarter2, y_quarter2])
    data_split.append([x_quarter3, y_quarter3])
    data_split.append([x_quarter4, y_quarter4])

    for k in range(len(lambdas)):
        valids = np.zeros(4)
        trainings = np.zeros(4)

        for i in range(4):
            x_valid = data_split[i][0]
            y_valid = data_split[i][1]
            current_x_train = []
            current_y_train = []
            for j in range(4):
                if j != i:
                    if len(current_x_train) == 0:
                        current_x_train = data_split[j][0]
                    else:
                        current_x_train = np.concatenate([current_x_train, data_split[j][0]])
                        
                    if len(current_y_train) == 0:
                        current_y_train = data_split[j][1]
                    else:
                        current_y_train = np.concatenate([current_y_train, data_split[j][1]])
                                         
         
            beta_train = normal_equation(current_x_train, current_y_train, lambdas[k])
            trainings[i] = get_loss(current_y_train, predict(current_x_train, beta_train))
            valids[i] = get_loss(y_valid, predict(x_valid, beta_train))
            
        
        valid_losses[k] = np.mean(valids)
        training_losses[k] = np.mean(trainings)

    return valid_losses, training_losses

    
# Calcuate the l2 norm of a vector    
def l2norm(vec):
    return np.linalg.norm(vec)

#  show the learnt values of Î² vector from the best Î»
def bar_plot(beta):
    i = np.linspace(0, beta.shape[0], beta.shape[0])
    plt.bar(i,beta)    
    plt.title("Bar Plot of Coefficients for the Best Beta")
    plt.xlabel('i')
    plt.ylabel('Beta_i')
    plt.show()
    

if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss") 
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]
    
    # step 2: analysis 
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = l2norm(normal_beta)# your code get l2 norm of normal_beta
    best_beta_norm = l2norm(best_beta)# your code get l2 norm of best_beta
    large_lambda_norm = l2norm(large_lambda_beta)# your code get l2 norm of large_lambda_beta

    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    
    
    # step 3: visualization
    bar_plot(best_beta)


    

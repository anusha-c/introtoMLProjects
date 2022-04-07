# Machine Learning HW1

import matplotlib.pyplot as plt
import numpy as np
# more imports

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    dataset = np.loadtxt(filename);
    y = dataset[:,-1];
    x = dataset[:, 0:-1]
    return x, y

# Find theta using the normal equation
def normal_equation(x, y):
    x_T = np.transpose(x)
    x_T_times_x = np.matmul(x_T, x)
    x_T_times_y = np.matmul(x_T, y)
    theta = np.matmul(np.linalg.inv(x_T_times_x), x_T_times_y)
    return theta

# Find thetas using stochastic gradient descent
# Don't forget to shuffle
def stochastic_gradient_descent(x, y, learning_rate, num_epoch):
     thetas = np.zeros([num_epoch, x.shape[1]])
     
     for i in range(1, num_epoch):
        a = np.copy(x)
        b = np.copy(y).reshape([y.shape[0],1])
        y_and_0 = np.append(b, np.zeros([a.shape[0],1]), axis=1)
        x_and_y = np.append(a, y_and_0, axis=1)
        np.random.shuffle(x_and_y)
        shuffled_x = x_and_y[:,0:-2]
        shuffled_y = x_and_y[:,-2]
        new_x = shuffled_x
        new_y = shuffled_y
        y_diff = new_y[0] - np.matmul(new_x[0],thetas[i-1])
        np.reshape(y_diff, [1,])
        thetas[i] = thetas[i-1] + learning_rate*(np.dot(y_diff, new_x[0]))  
     return thetas

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_epoch):
    thetas1 = np.zeros(num_epoch*x.shape[1])
    thetas1 = np.reshape(thetas1, [num_epoch, x.shape[1]])
    x_T = np.transpose(x)
    
    for i in range(1,num_epoch):
        x_times_theta = np.matmul(x, thetas1[i-1,:])
        x_times_theta = np.reshape(x_times_theta, [x.shape[0],])
        y_diff = y - x_times_theta
        thetas1[i, :] = thetas1[i-1, :] + learning_rate*np.matmul(x_T, y_diff)
    return thetas1

# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_epoch, batch_size):
    x, y = load_data_set('regression-data.txt')
    thetas = np.zeros([num_epoch, x.shape[1]])
    
    for i in range(num_epoch):
        a = np.copy(x)
        b = np.copy(y).reshape([y.shape[0],1])
        y_and_0 = np.append(b, np.zeros([a.shape[0],1]), axis=1)
        x_and_y = np.append(a, y_and_0, axis=1)
        np.random.shuffle(x_and_y)
        shuffled_x = x_and_y[:,0:-2]
        shuffled_y = x_and_y[:,-2]
        new_x = shuffled_x[0:batch_size,:]
        new_y = shuffled_y[0:batch_size]
        new_x_T = np.transpose(new_x)
        
        x_times_theta = np.matmul(new_x, thetas[i-1,:])
        x_times_theta = np.reshape(x_times_theta, [new_x.shape[0],])
        y_diff = new_y - x_times_theta
        thetas[i, :] = thetas[i-1, :] + learning_rate*np.matmul(new_x_T, y_diff)
        
    return thetas

# Given an array of x and theta predict y
def predict(x, theta):
   y_predict = np.matmul(x, theta)
   return y_predict

# Given an array of y and y_predict return MSE loss
def get_mseloss(y, y_predict):
    y_diff_squared = (y-y_predict)**2
    loss = np.mean(y_diff_squared)
    return loss

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, thetas, title):
    losses = []
    epochs = []
    losses = []
    epoch_num = 1
    for theta in thetas:
        losses.append(get_mseloss(y, predict(x, theta)))
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # first column in data represents the intercept term, second is the x value, third column is y value
    x, y = load_data_set('regression-data.txt')
    
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")
    

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    # thetas records the history of (s)GD optimization e.g. thetas[epoch] with epoch=0,1,,......T
    
    thetas = gradient_descent(x, y, 0.00003, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss")
    
    thetas = gradient_descent(x, y, 0.009, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss")
    
    thetas = gradient_descent(x, y, 0.003, 100) 
    plot(x, y, thetas[-1], "Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Gradient Descent Epoch vs Mean Training Loss")
    
    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    
    thetas = stochastic_gradient_descent(x, y, 0.03, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit")
    
    thetas = stochastic_gradient_descent(x, y, 2.5, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit")
    
    thetas = stochastic_gradient_descent(x, y, 0.2, 100) # Try different learning rates and number of epoch
    plot(x, y, thetas[-1], "stochastic Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Stochastic Gradient Descent Epoch vs Mean Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of epoch
    thetas = minibatch_gradient_descent(x, y, 0.002, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")
    
    thetas = minibatch_gradient_descent(x, y, 0.1, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")
    
    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 20)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")
    
    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 100)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")
                         
    thetas = minibatch_gradient_descent(x, y, 0.02, 100, 35)
    plot(x, y, thetas[-1], "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, thetas, "Minibatch Gradient Descent Epoch vs Mean Training Loss")

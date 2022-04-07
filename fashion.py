
# Machine Learning Homework 4 - Image Classification

# General imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import sys
import pandas as pd

# Keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Softmax
from keras.wrappers.scikit_learn import KerasClassifier


### Already implemented
def get_data(datafile):
    dataframe = pd.read_csv(datafile)
    data = list(dataframe.values)
    labels, images = [], []
    for line in data:
        labels.append(line[0])
        images.append(line[1:])
    labels = np.array(labels)
    images = np.array(images).astype('float32')
    images /= 255
    return images, labels


### Already implemented
def visualize_weights(trained_model, num_to_display=20, save=True, hot=True):
    layer1 = trained_model.layers[0]
    weights = layer1.get_weights()[0]
   
    # Feel free to change the color scheme
    colors = 'hot' if hot else 'binary'
    try:
        os.mkdir('weight_visualizations')
    except FileExistsError:
        pass
    for i in range(num_to_display):
        wi = weights[:,i].reshape(28, 28)
        plt.imshow(wi, cmap=colors, interpolation='nearest')
        if save:
            plt.savefig('./weight_visualizations/unit' + str(i) + '_weights.png')
        else:
            plt.show()


### Already implemented
def output_predictions(predictions, model_type):
    if model_type == 'CNN':
        with open('CNNpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')
    if model_type == 'MLP':
        with open('MLPpredictions.txt', 'w+') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')


def plot_history(history):
    train_loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    train_acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    num_epochs = 11
    # plot
    train_loss = plt.figure(1)
    plt.plot( range(1,(num_epochs+1)), train_loss_history)
    plt.title("Training Loss vs Epoch")

    val_loss = plt.figure(2)
    plt.plot( range(1,(num_epochs+1)), val_loss_history)
    plt.title("Validation Loss vs Epoch")
   
    train_acc = plt.figure(3)
    plt.plot( range(1,(num_epochs+1)), train_acc_history)
    plt.title("Training Accuracy vs Epoch")
  
    val_acc = plt.figure(4)
    plt.plot( range(1,(num_epochs+1)), val_acc_history)
    plt.title("Validation Accuracy vs Epoch")

    plt.show(block=True)
    
    


num_classes = 10

def create_mlp(args=None):
	# You can use args to pass parameter values to this method

	# Define model architecture
	model = Sequential()
	model.add(Dense(units=128, activation='relu', input_dim=28*28))
	model.add(Dropout(0.3))
	model.add(Dense(units=num_classes, activation='softmax'))
	# add more layers...

	# Optimizer
	if args['opt'] == 'sgd':
			optimizer = keras.optimizers.SGD(lr=args['learning_rate'])
	elif args['opt'] == 'adam':
			optimizer = keras.optimizers.Adam(lr=args['learning_rate'])
	 
	# Compile
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	return model
		

def train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=None):
    # You can use args to pass parameter values to this method
    y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
    model = create_mlp(args)
    history = model.fit(x_train, y_train, batch_size=args['batch_size'], epochs=args['epoch'], validation_split=args["validation_split"])
    return model, history


def create_cnn(args=None):
    # You can use args to pass parameter values to this method

    # 28x28 images with 1 color channel
		input_shape = (28, 28, 1)
		num_classes = 10

    # Define model architecture
    
		model = Sequential()
		model.add(Conv2D(filters=256, activation='relu', kernel_size=(2,1), strides=1, input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(2,1), strides=1))
		model.add(Conv2D(filters=256, activation='relu', kernel_size=(1,2), strides=1, input_shape=input_shape))
		model.add(MaxPooling2D(pool_size=(1,2), strides=1))
		model.add(Conv2D(filters=128, activation='relu', kernel_size=(2,2), strides=1))
		model.add(MaxPooling2D(pool_size=(2,2), strides=1))
		# can add more layers here...
		model.add(Flatten())
		# can add more layers here...
		model.add(Dense(units=128, activation='relu'))
		model.add(Dense(units=num_classes, activation='softmax'))

    # Optimizer
		if args['opt'] == 'sgd':
				optimizer = keras.optimizers.SGD(lr=args['learning_rate'])
		elif args['opt'] == 'adam':
				optimizer = keras.optimizers.Adam(lr=args['learning_rate'])

    # Compile
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		print(model.summary())
		return model

def train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=None):
	# You can use args to pass parameter values to this method
		x_train = x_train.reshape(-1, 28, 28, 1)
		y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
		model = create_cnn(args)
		history = model.fit(x_train, y_train, batch_size=args['batch_size'], epochs=args['epoch'], validation_split=args["validation_split"])
		return model, history



def train_and_select_model(train_csv, model_type, grading_mode):
    """Optional method. You can write code here to perform a 
    parameter search, cross-validation, etc. """
    num_epochs = 11
    x_train, y_train = get_data(train_csv)

    args = {
        'batch_size': 128,
        'validation_split': 0.1,
				'epoch': num_epochs
    }
    
    best_valid_acc = 0
    best_hyper_set = {}
    other_hyper_set = [64]
    
    ## Select best values for hyperparamters such as learning_rate, optimizer, hidden_layer, hidden_dim, regularization...
   
    if not grading_mode:
        for learning_rate in [0.002]:
            for opt in ['adam']:
                for other_hyper in other_hyper_set:  ## search over other hyperparameters
                  args['opt'] = opt
                  args['learning_rate'] = learning_rate
                  args['other_hyper'] = other_hyper

									
                  if model_type == 'MLP':
                      model, history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
                  else:
                      model, history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args)

                  validation_accuracy = history.history['val_accuracy']

                  max_valid_acc = max(validation_accuracy)
                  if max_valid_acc > best_valid_acc:
                      best_model = model
                      best_valid_acc = max_valid_acc
                      best_hyper_set['learning_rate'] = learning_rate
                      best_hyper_set['opt'] = opt
                      best_history = history
    else:
        ## In grading mode, use best hyperparameters you found 
        if model_type == 'MLP':
            args['opt'] = 'adam'
            args['learning_rate'] = 0.002
            
            best_model, best_history = train_mlp(x_train, y_train, x_vali=None, y_vali=None, args=args)
        
        if model_type == 'CNN':
            args['opt'] = 'adam'
            args['learning_rate'] = 0.002
            best_model, best_history = train_cnn(x_train, y_train, x_vali=None, y_vali=None, args=args) 

    return best_model, best_history


def logpca(train_file):
    features = [10, 100, 500, 784] 
    x_train, y_train = get_data(train_file)
    scaler = StandardScaler()
    scaler.fit(x_train)
    scaled_data = scaler.transform(x_train)
    all_pca_predictions = []
    all_pca_scores = []
    
    x_train_logreg, x_val_logreg, y_train_logreg, y_val_logreg = train_test_split(x_train[0:20000,:], y_train[0:20000], test_size=0.1)
    
    for f in features:
      pca = PCA(n_components = f)
      pca.fit(scaled_data)
      x_train_pca = pca.transform(scaled_data)
    
      x_train_pca, x_val_pca, y_train_pca, y_val_pca = train_test_split(x_train_pca, y_train, test_size=0.1)
      logreg = LogisticRegression(solver='sag', max_iter=10000, multi_class='multinomial')
      logreg.fit(x_train_pca, y_train_pca)
      log_pca_predictions = logreg.predict(x_val_pca)
      log_pca_score = 1 - logreg.score(x_val_pca, y_val_pca)
      all_pca_predictions.append(log_pca_predictions)
      all_pca_scores.append(log_pca_score)
  
    plt.plot( features, all_pca_scores)
    plt.title("Average PCA error vs number of features selected")
    plt.show(block=True)
  
    print(all_pca_predictions, all_pca_scores)
    return all_pca_predictions, all_pca_scores


if __name__ == '__main__':
      ### Switch to "grading_mode = True" before you submit ###
      grading_mode = True
      if grading_mode:
        # When we grade, we'll provide the file names as command-line arguments
        if (len(sys.argv) != 3):
            print("Usage:\n\tpython3 fashion.py train_file test_file")
            exit()
        train_file, test_file = sys.argv[1], sys.argv[2]
        
        # train your best model
        best_mlp_model, _ = train_and_select_model(train_file, model_type='MLP', grading_mode=True)
        
        x_test, _ = get_data(test_file)
        # use your best model to generate predictions for the test_file
        mlp_pred = best_mlp_model.predict(x_test)
        mlp_predictions = np.argmax(mlp_pred, axis=1)
        output_predictions(mlp_predictions, model_type='MLP')
        
      
        x_test = x_test.reshape(-1, 28, 28, 1)
        best_cnn_model, _ = train_and_select_model(train_file, model_type='CNN', grading_mode=True)
        cnn_pred = best_cnn_model.predict(x_test)
        cnn_predictions = np.argmax(cnn_pred, axis=1)
        output_predictions(cnn_predictions, model_type='CNN')
        
        with open('predictions.txt', 'w+') as f:
            for pred in cnn_predictions:
                f.write(str(pred) + '\n')
                
        # pca_predictions = logpca(train_file)

      
      else:
        ### Edit the following two lines if your paths are different
        train_file = '/content/drive/My Drive/HW3_COLAB/fashion_data2student/fashion_train.csv'
        test_file = '/content/drive/My Drive/HW3_COLAB/fashion_data2student/fashion_test.csv'
        # # MLP
        # mlp_model, mlp_history = train_and_select_model(train_file, model_type='MLP', grading_mode=False)
        # plot_history(mlp_history)
        # visualize_weights(mlp_model)
          
        # # CNN
        # cnn_model, cnn_history = train_and_select_model(train_file, model_type='CNN', grading_mode=False)
        # plot_history(cnn_history)
        
       
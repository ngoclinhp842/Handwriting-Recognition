'''digitReg.py
MICHELLE PHAN

How to run: python digitReg.py


Train a RBF model network on "real" image dataset of handwritten number digits
to correctly predict the numeric digit in an image:

- 60,000 images in training set, 10,000 images in test set.
- Each image is 28x28 pixels.
- The images are grayscale (no RGB colors).
- Each image (data sample) contains ONE of 10 numeric digit 0, 1, 2, ... 8, 9.

More information about MNIST: http://yann.lecun.com/exdb/mnist/
'''
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from rbf_net import RBF_Net

plt.style.use(['seaborn-v0_8-colorblind', 'seaborn-v0_8-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)

def main():
    x_train = np.load('data/mnist_train_data.npy')
    x_test = np.load('data/mnist_test_data.npy')
    y_train = np.load('data/mnist_train_labels.npy')
    y_test = np.load('data/mnist_test_labels.npy')
    
    print(f'Your training set shape is {x_train.shape} and should be (60000, 28, 28).')
    print(f'Your training classes shape is {y_train.shape} and should be (60000,).')
    print(f'Your test set shape is {x_test.shape} and should be (10000, 28, 28).')
    print(f'Your test classes shape is {y_test.shape} and should be (10000,).')
    
    # showing the first 25 images in the dataset
#    imgs = x_train[:25]
#    img_plot(imgs)
    
    # preprocess data
    x_train, x_test = preprocess_data(x_train, x_test)
    
    train_and_access(x_train, y_train, x_test, y_test)
        
def train_and_access(x_train, y_train, x_test, y_test):
    # get the current time in seconds since the epoch
    seconds = time.process_time()

    mnist_net = RBF_Net(200, 10)

    mn_train_set = x_train[:2000]
    y_train_set = y_train[:2000]

    mnist_net.train(mn_train_set, y_train_set)
    mnist_net.train(mn_train_set, y_train_set)
    print("Training RBF without PCA took " + str(seconds) + ' seconds')
    
    # train acc
    y_pred_train = mnist_net.predict(mn_train_set)
    acc_train = mnist_net.accuracy(y_train_set, y_pred_train)
    print('Accuracy on the training set: {}% '.format(acc_train * 100))
    
    # test acc
    y_pred_test = mnist_net.predict(x_test)
    acc_test = mnist_net.accuracy(y_test, y_pred_test)
    print('Accuracy on the testing set: {}% '.format(acc_test * 100))
    
    # Visualize network hidden layer prototypes
    prototypes = mnist_net.get_prototypes()
    prototypes = np.reshape(prototypes, [prototypes.shape[0], 28, 28])

    cols = rows = 5
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(prototypes[i*rows + j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    plt.show()
    

# 5x5 grid showing the first 25 images in the dataset
def img_plot(imgs):
    '''Create a 5x5 grid of face images
    
    Parameters:
    -----------
    face_imgs: ndarray. shape=(N, img_y, img_x).
        Grayscale images to show.
    
    TODO:
    - Create a 5x5 grid of plots of a legible size
    - In each plot, show the grayscale image and make the title the person's name.
    '''
    # Create two subplots and unpack the output array immediately
    fig, axs = plt.subplots(5, 5)
    
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(imgs[5 * i + j], cmap=plt.get_cmap('gray'))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    fig.set_size_inches(6, 6)
    fig.suptitle('The first 25 images in the dataset')
    fig.tight_layout()
    plt.show()
    
def preprocess_data(x_train, x_test):
    # flatten train and test sets dimensions
    M = x_train.shape[1]
    N = x_train.shape[0]
    x_train = np.reshape(x_train, (N, M * M))
    print('Fattening x_train: ', x_train.shape)

    M = x_test.shape[1]
    N = x_test.shape[0]
    x_test = np.reshape(x_test, (N, M * M))
    print('Fattening x_test: ', x_test.shape)

    # normalize so that max value in each image is 1 by dividing by 255
    x_train = x_train /255
    x_test = x_test /255

    print('Max value in x_train after normalization: ', x_train.max())
    print('Max value in x_test after normalization: ', x_test.max())
    return x_train, x_test
    

if __name__ == "__main__":
   main()

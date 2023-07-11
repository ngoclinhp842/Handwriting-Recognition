# Handwriting-Recognition
An <a href="https://towardsdatascience.com/radial-basis-function-neural-network-simplified-6f26e3d5e04d">RBF</a> neural network that recognizes which human handwritten digit (i.e. the numbers 0, 1, ..., 9) is shown in <a href="http://yann.lecun.com/exdb/mnist/">MNIST</a> image dataset with 91.5% accuracy.

## 🚀 Running Handwriting-Recognition:
To run digitReg.py

```sh
python3 digitReg.py
```

## ☘️ Description 
Improve performance on MNIST with PCA

Using all 768 features (pixels) in each image may not be very helpful for classification. For example, pixels around the border are almost always white. Transform the dataset(s) using PCA to compress the number of features before training your RBF network. Experiment with PCA to improve classification accuracy and runtime performance.

## 👀 Results:
* Training RBF without PCA took 4.576543 seconds

Accuracy on the training set: 94.55% 

Accuracy on the testing set: 91.3% 

* Pca took 99.737493 seconds

Training took 134.519913 seconds

Accuracy on the training set: 95.0% 

Accuracy on the testing set: 10.83% 

## 📝 Analysis:

## 

## Finding
1. Record the time it took to train RBF without PCA. I add time.process_time() from time module to record the training time

        Training RBF without PCA took 1556.191151 seconds
        Number of training samples: 2000
        Number of hidden units: 200
        Accuracy on the training set: 95.1% 
        Accuracy on the testing set: 91.56% 

2. Perform PCA and record time and its performance.

Algorithm: Keeping the same number of training samples and hidden units

- Preprocess data: Reshape x_train and x_test into (M, N * N) shape
- Perform PCA with perform_pca function to keep 150 PCs (I choose this number after looking at the elbow plot and see that under 200 top PCs account for 90% of the data variance)
        
        Pca took 1674.91287 seconds
        
- Record the time it took to train RBF after PCA

        Training RBF after PCA took 1725.963101 seconds
        Accuracy on the training set: 94.1% 
        Accuracy on the testing set: 10.36% 
        
## Analysis

PCA is useful for reducing the dimensionality of high-dimensional data by projecting it onto a lower-dimensional space while preserving most of the variance. This can help to remove redundant or noisy features from the input data, which can improve the performance of the RBF network by reducing overfitting and increasing the network's ability to generalize to new data.

However, if the input data already has a relatively low dimensionality and does not contain a lot of noise or redundant features, applying PCA may not be necessary and can even lead to a loss of information. In such cases, it may be better to use the original input data directly without applying PCA.

MNIST is the later case because the data is already relatively low dimensionality and does not have a lot of noise. Applying PCA actually lead to loss of information that helps make prediction for new input. This explains why the accuracy on the testing set is so low (10.05%) while the accuracy on the training set is still very high (92.45%)

For time performace, if we combine the time to run PCA and RBF together, running PCA actually costs more than in this case. 

'''rbf_net.py
Radial Basis Function Neural Network
MICHELLE PHAN
CS 252: Mathematical Data Analysis Visualization, Spring 2021
'''
import numpy as np
import kmeans
import linear_regression as lr


class RBF_Net:
    def __init__(self, num_hidden_units, num_classes):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''
        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None
        
        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None
        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None
        
        # k: number of hidden units as an instance variable
        self.k = num_hidden_units
        
        # num_classes: number of classes (number of output units in network) as an instance variable
        self.num_classes = num_classes

    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes
    
    def get_wts(self):
        '''Returns the wts

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(num_hidden_units+1, num_classes)
        '''
        return self.wts

    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_classes

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        avg_dists = np.zeros((self.k,))
        
        # loop through each sample in each class
        for i in range(centroids.shape[0]):
            # calculate the distance between each cluster center and corresponding data points
            # take the average of all those distance
            avg_dists[i] = np.mean(np.linalg.norm(data[cluster_assignments == i] - centroids[i], axis = 1))
        return avg_dists
        

    def initialize(self, data):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        # run kmeans with k = self.k
        kmeansObj = kmeans.KMeans(data)
        kmeansObj.cluster_batch(k=self.k, n_iter=5)
        
        # set prototypes instance variable
        self.prototypes = kmeansObj.get_centroids()
        
        cluster_assignments = kmeansObj.get_data_centroid_labels()
        
        # set sigmas instance variable
        self.sigmas = self.avg_cluster_dist(data, self.prototypes, cluster_assignments, kmeansObj)
        

    def linear_regression(self, A, y):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        li_reg = lr.LinearRegression(A)
        return li_reg.linear_regression_qr(A,y)

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        activation = np.zeros((data.shape[0], self.k))
        
        # loop through each sample in each class
        for i in range(self.k):
            dist = np.linalg.norm(data - self.prototypes[i], axis = 1)
            activation[:,i] = np.exp((-1 * dist **2) / (2 * self.sigmas[i]**2 + 1e-8))
        
        return activation
        

    def output_act(self, hidden_acts):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        
        '''
        # add a column of 1 at the right hand of hidden_acts
        z = np.ones((hidden_acts.shape[0], 1))
        new_hidden_acts = np.hstack((hidden_acts, z))
        # Multiply hidden unit activation by output unit weights.
        return new_hidden_acts @ self.wts

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        # initialize the network
        net = self.initialize(data)
        classes = np.arange(self.num_classes)
        
        # caculate the hiden layer activations
        hidden_acts = self.hidden_act(data)
        
        # instance variable to store the weights between the hidden and output layer weights
        self.wts = np.zeros((self.k + 1, self.num_classes))
        
        for i in range(self.num_classes):
            # construct a matrix to indicate which claas is matched
            match_y = np.array(classes[i] == y).astype(int)
            # caculate the activation of the hidden layer units
            self.wts[:,i] = self.linear_regression(hidden_acts, match_y)
            

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        # ndarray to store class prediction
        y_pred = np.zeros((data.shape[0],))
        # caculate the hidden layer's activations
        hidden_acts = self.hidden_act(data)
        # caculate the output layer's activations
        output_acts = self.output_act(hidden_acts)
        
        # for each data sample,
        for i in range(data.shape[0]):
            # assigned class: index of unit with largest activation
            y_pred[i] = np.argmax(output_acts[i])
        return y_pred

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        # count how many  the output labels are correct (y == y_pred)
        correct = np.sum(y == y_pred)
        # divide that by total number of labels to get the probability
        acc = correct / y.shape[0]
        return acc


class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''
    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)
        self.h_sigma_gain = h_sigma_gain

    def hidden_act(self, data):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''
        activation = np.zeros((data.shape[0], self.k))
        
        # loop through each sample in each class
        for i in range(self.k):
            dist = np.linalg.norm(data - self.prototypes[i], axis = 1)
            activation[:,i] = np.exp((-1 * dist **2) / (2 * self.h_sigma_gain * self.sigmas[i]**2 + 1e-8))
        
        return activation

    def train(self, data, y):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''
        # initialize the network
        net = self.initialize(data)
        classes = np.arange(self.num_classes)
        
        # caculate the hiden layer activations
        hidden_acts = self.hidden_act(data)
        
        # instance variable to store the weights between the hidden and output layer weights
        self.wts = np.zeros((self.k + 1, self.num_classes))
        
        for i in range(self.num_classes):
            # caculate the activation of the hidden layer units
            self.wts[:,i] = self.linear_regression(hidden_acts, y)

    def predict(self, data):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''
        # caculate the hidden layer's activations
        hidden_acts = self.hidden_act(data)
        # caculate the output layer's activations
        output_acts = self.output_act(hidden_acts)
        return output_acts
    
    def compute_mse(self, y, y_pred):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error
        '''
        # MSE = sum of residuals squared / number of samples
        res = y - y_pred
        # E = sum of residuals squared
        E = np.sum(np.square(res))
        # number of samples
        N = y.shape[0]
        return E/N
'''kmeans.py
Performs K-Means clustering
Michelle Phan
CS 252: Mathematical Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors


class KMeans:
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps = self.data.shape[0]
        self.num_features = self.data.shape[1]
        
    def set_centroids(self, centroids):
        '''Replaces data instance variable with `centroids`.

        Parameters:
        -----------
        data: ndarray. shape=(k, self.num_features)
        '''
        self.centroids = centroids

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return np.copy(self.data)

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(pt_1 - pt_2)

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        return np.linalg.norm(pt - centroids, axis = 1)

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        # get number of samples 
        self.num_samps = self.data.shape[0]
        self.num_features = self.data.shape[1]
        
        centroids = self.data[np.random.choice(self.num_samps, k, replace=False)]
        
        self.k = k 
        self.centroids = np.array(centroids)
        return self.centroids

    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''
        self.k = k
        # Choose first centroid randomly
        centroids = np.zeros((k, self.num_features))
        centroids[0] = self.data[np.random.choice(self.num_samps)]
        
        # Choose remaining centroids
        for i in range(1, k):
            # compute distance between all remaining samples and the 1st centroid
            distances = np.zeros((self.num_samps, i))
            for j in range(i):
                distances[:, j] = np.sum((self.data - centroids[j]) ** 2, axis=1)

            # Choose next centroid with probability proportional to distance to nearest centroid
            min_distances = np.min(distances, axis=1)
            probs = min_distances / np.sum(min_distances)
            centroids[i] = self.data[np.random.choice(self.num_samps, p=probs)]

        return centroids
      

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, init_method='random'):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        # Initialize K-means variables
        if init_method == 'random':
            prev_centroids = self.initialize(k)
        elif init_method == 'kmeans++':
            prev_centroids = self.initialize_plusplus(k)
            
        
        labels = self.update_labels(prev_centroids)
        self.centroids, centroid_diff = self.update_centroids(k, labels, prev_centroids)
        
        # number of iterations K-means run for 
        iteration = 1

        
        # do k-means as long as max_iter is not meet AND abs(prev_centroids - self.centroids) > tol
        while np.any(abs(self.centroids - prev_centroids) > tol):
            if (iteration >= max_iter):
                break
            labels = self.update_labels(self.centroids)
            prev_centroids = self.centroids
            self.centroids, centroid_diff = self.update_centroids(k, labels, prev_centroids)
            iteration += 1
            
        self.k = k
        self.data_centroid_labels = labels
        self.inertia = self.compute_inertia()
        
        if verbose:
            print('Total number of iterations K-means ran for is', iteration)
        return self.inertia, iteration
        

    def cluster_batch(self, k=2, n_iter=1, verbose=False, init_method='random'):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        inertia_list = []
        centroids_list = []
        labels = []
        interations = []
        for _ in range(n_iter):
            if init_method =='random':
                inertia, iteration = self.cluster(k = k, init_method ='random')
            else:
                inertia, iteration = self.cluster(k = k, init_method ='kmeans++')
            inertia_list.append(inertia)
            centroids_list.append(self.centroids)
            labels.append(self.data_centroid_labels)
            interations.append(iteration)
            
        self.inertia = inertia_list[np.argmin(inertia_list)]
        self.centroids = centroids_list[np.argmin(inertia_list)]
        self.data_centroid_labels = labels[np.argmin(inertia_list)]
        return self.inertia, np.mean(interations)
        

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = []
        for i in range(self.num_samps):
            dists = self.dist_pt_to_centroids(self.data[i], centroids)
            centroid_idx = np.argmin(dists, axis = 0)
            labels.append(centroid_idx)
        labels = np.array(labels)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        if self.data.dtype == 'complex':
            new_centroids = np.zeros((k, self.num_features), dtype='complex')
        else:
            new_centroids = np.zeros((k, self.num_features))
        for cluster in range(k):
            is_cluster = data_centroid_labels == cluster
            # no cluster is assigned to a correct label
            if not np.any(is_cluster):
                # randomly pick a sample in self.data to be new centroid
                new_centroids[cluster] = self.data[np.random.choice(np.arange(self.num_samps))]
            # else, update centroids to be the mean of all samples in 1 label
            else:
                new_centroids[cluster] = np.mean(self.data[is_cluster], axis = 0)
            
        centroid_diff = new_centroids - prev_centroids
        self.centroids = new_centroids
        self.data_centroid_labels = data_centroid_labels
        self.k = k
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        # compute distance between every data sample and its assigned (nearest) centroid
        dist_sum = 0
        for cluster in range(self.centroids.shape[0]):
            is_cluster = self.data_centroid_labels == cluster
            dists = np.linalg.norm(self.data[is_cluster] - self.centroids[cluster], axis = 1)
            dist_sum += np.sum(dists ** 2)
        return dist_sum / self.num_samps

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        # Define ColorBrewer color map palette with 10 colors
        # as a list of RGB tuples
        brewer_colors = cartocolors.qualitative.Safe_10.mpl_colors
        fig, axs = plt.subplots(1, 1)
        
        # Create two subplots and unpack the output array immediately
        for cluster in range(self.k):
            is_cluster = self.data_centroid_labels == cluster
            
            w = self.data[is_cluster]
            axs.scatter(w[:,0], w[:,1], color = brewer_colors[cluster])
        
        # plot the centroids
        for i in range(self.k):
            axs.plot(self.centroids[i, :][0], self.centroids[i, :][1], 'k*')

        plt.legend()
        plt.tight_layout()

    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []
        x = np.arange(1, max_k + 1)
        for i in range(1, max_k + 1):
            kmean = self.cluster_batch(k = i)
            inertias.append(kmean)
        
        # elbow_plot with inertia of k-means with k=1,2,...,max_k
        f = plt.figure(figsize = (8, 8))
        plt.plot(x, inertias, marker = '.', markersize = 12)
        plt.xlabel('Number of cluster')
        plt.ylabel('Inertia')
        plt.title('Elbow plot')
            

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        for cen in range(self.k):
            self.data.setflags(write=1)
            self.data[self.data_centroid_labels == cen] = self.centroids[cen]
            
            

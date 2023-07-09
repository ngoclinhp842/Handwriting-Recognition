'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
YOUR NAME HERE
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None
        
        # p matrix that project A into PCAs space. shape = (M,top_k)
        self.p = None

        self.orig_min = None
        self.orig_max = None
        self.orig_mean = None

    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs
    
    def get_p(self):
        '''(No changes should be needed)'''
        return self.p
    
    def get_A(self):
        '''(No changes should be needed)'''
        return np.copy(self.A)

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        (N,M) = data.shape
        cov = np.ones((M,M))
        mean = data.mean(0)
        cen = data - mean
        cov = cen.T @ cen/(N-1)
        return cov

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        e_vals_sum = np.sum(e_vals)
        return (e_vals/e_vals_sum).tolist()
        

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        cum_var = np.ones(len(prop_var), dtype = 'complex_')
        cur_val = 0
        for i in range (len(prop_var)):
            cur_val += prop_var[i]
            cum_var[i] = cur_val
        prop_var_sum = np.sum(prop_var)
        return (cum_var/prop_var_sum).tolist()
            
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        # revelant data
        self.A = np.array(self.data[vars])
        self.vars = vars
        self.normalized = normalize
        
        if normalize and self.A.shape[0] > 1:
            # needed to "undo" or reverse the normalization on the selected data
            pre_A = self.A
            # normalize the selected data (seperately, based on dynamic range of each variable)
            cur_range = self.A.max(0) - self.A.min(0)
            self.orig_min = self.A.min(0)
            self.orig_max = self.A.max(0)
            self.orig_mean = self.A.mean(0)
            self.A = (self.A - self.A.min(0)) / cur_range
            
        if self.A.shape[0] > 1:
            cov = self.covariance_matrix(self.A)
            # calculate all instance variables
            egen = np.linalg.eig(cov)
            # set eigenvalues
            self.e_vals = egen[0]
            # set eigenvectors
            self.e_vecs = egen[1]
            self.prop_var = self.compute_prop_var(self.e_vals)
            self.cum_var = self.compute_cum_var(self.e_vals)
        
        

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        # proportion variance accounted for
        y = self.compute_cum_var(self.prop_var)
        
        if num_pcs_to_keep != None:
            # top PCs included (large-to-small order)
            x = np.arange(num_pcs_to_keep)
        else:
            x = np.arange(1, self.A.shape[1] + 1)
        # show variance accounted for by ALL the PCs (the default)
        f = plt.figure(figsize = (8, 8))
        plt.plot(x, y, marker = '.', markersize = 12)
        plt.xlabel('Top PCs included')
        plt.ylabel('Proportion variance')
        plt.title('Elbow plot')
        
           

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        # compute center data 
        A_cen = self.A - self.A.mean(0)
        # compute rotation matrix = matrix of eigenvectors
        e_val, p = np.linalg.eig(self.covariance_matrix(self.A))
        
        # update p to drop PCs
        p = p[:,:pcs_to_keep]
        # project data into PCA space
        self.A_proj = A_cen @ p
        self.p = p
        return self.A_proj

    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        self.A_proj = self.pca_project(np.arange(top_k))
        
        # compute rotation matrix
        e_val, e_vec = np.linalg.eig(self.covariance_matrix(self.A))
        p = e_vec
        
        # update p to drop PCs
        p = p[:, :top_k]
       
        # if didn't normalized
        if self.normalized == False:
            A_re = (self.A_proj @ p.T) + self.A.mean(0)
        else:
            A_re = (self.orig_max - self.orig_min)*(self.A_proj @ p.T) + self.orig_mean
            
        return A_re

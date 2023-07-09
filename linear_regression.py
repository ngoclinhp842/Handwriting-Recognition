'''linear_regression.py
Subclass of Analysis that performs linear regression on data
MICHELLE PHAN
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import scipy.linalg as li
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean squared error (MSE). float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, method='scipy', p=1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization
        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        # select the variable columns associated with the independent and dependent variable strings
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        
        # Calculate the linear regression with the given method
        if method == 'scipy':
            p = self.linear_regression_scipy(self.A,self.y)
        elif method == 'normal':
            p = self.linear_regression_normal(self.A,self.y)
        elif method == 'qr':
            p = self.linear_regression_qr(self.A, self.y)

        # Last item in p (right most item) is the intercept
        # Remaning iterms are slopes
        self.slope = p[:len(p) - 1]
        self.slope = np.reshape(self.slope, (self.slope.shape[0], 1))
        self.intercept = p[len(p) - 1][0]

        # calculate the residuals
        y_pred = self.predict()
        res = self.compute_residuals(y_pred)
        self.residuals = res

        # calculate and print the R^2 value
        self.R2 = self.r_squared(y_pred)

        # set all instance variables in constructor
        self.ind_vars = ind_vars
        self.dep_var = dep_var

        # calculate mean squared errors
        self.mse = self.compute_mse()



    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,1)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        # adding homogeneous coordinates (for the intercept) to the A matrix
        h = np.ones((A.shape[0],1))
        A = np.hstack((A, h))

        # Calculate the linear regression
        p = li.lstsq(A, y)

        # Last item in p[0] (right most item) is the intercept
        # Remaning iterms are slopes
        return p[0]

        

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,1)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''
        # adding homogeneous coordinates (for the intercept) to the A matrix
        h = np.ones((A.shape[0],1))
        A = np.hstack((A, h))
        
        # Last item in c (right most item) is the intercept
        # Remaning iterms are slopes
        A_inv = li.inv(A.T @ A)
        c = A_inv @ A.T @ y
        return c
        

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        # adding homogeneous coordinates (for the intercept) to the A matrix
        h = np.ones((A.shape[0],1))
        A = np.hstack((A, h))

        # R @ c = Q^T @ y
        Q, R = self.qr_decomposition(A)
        right = Q.T @ y
        c = li.solve_triangular(R, right)
        return c

        

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.

        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram-Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        # initialize Q matrix, shape = (num_data_samps, num_ind_vars+1)
        N = A.shape[0]
        M = A.shape[1]
        Q = np.copy(A)
        # pointer to current column
        for i in range(M):
            # make a copy of the current column
            cur = np.copy(A[:,i])
            # loop through  all columns to the left of column i
            for j in range(i):
                # subtract off overlap with all j columns (already orthogonal and normalized)
                cur -= ((Q[:,j] @ A[:,i]) * Q[:,j])
            # normalize the current column (make sure don't devide it by 0)
            Q[:,i] = cur / (np.linalg.norm(cur) + 1e-20)
        
        # calculate R 
        R = Q.T @ A
        return Q, R


    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        if X is None: 
            X = self.A

        # plug in data X in y_pred = mA + b (A = X, m are slopes, b is intercept)
        y_pred = self.slope.T @ X.T + self.intercept

        return y_pred.T

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        # R^2 = 1 - E/smd
        res = self.compute_residuals(y_pred)
        y_avg = np.mean(self.y)
        # E = sum of residuals squared
        E = np.sum(np.square(res))
        # smd  = sum of differences between y and y mean squared
        smd =np.sum(np.square(self.y - y_avg))
        return 1 - E/smd

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        return self.y - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        # MSE = sum of residuals squared / number of samples
        y_pred = self.predict()
        res = self.compute_residuals(y_pred)
        # E = sum of residuals squared
        E = np.sum(np.square(res))
        # number of samples
        N = self.y.shape[0]
        return E/N

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        # Use scatter() in Analysis to handle the plotting of points. 
        x,y = analysis.Analysis.scatter(self, ind_var, dep_var, title= title)

        # create x values to draw linear regression
        # (100 points equally spaced from x min to x max)
        line_x = np.linspace(np.min(x), np.max(x), 100)
        line_x_s = np.reshape(line_x, (line_x.shape[0], 1))
        
        if self.p > 1:
            line_x_s = self.make_polynomial_matrix(line_x, self.p) 

        line_y = self.predict(line_x_s)

        # plot the linear regression in red
        plt.plot(line_x,line_y, c='red')
        # if R2 is calculated, include it in the title
        if (self.R2 is not None):
            title = title + '\n$R^2$ = ' + str(round(self.R2, 3))

        # set corresponding title, labels, legend
        plt.title(title)
        plt.xlabel(str(ind_var))
        plt.ylabel(str(dep_var))

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        # Use pair_plot() in Analysis to take care of making the grid of scatter plots.
        fig, axs = analysis.Analysis.pair_plot(self, data_vars, fig_sz)

        # Create just a figure and only one subplot
        var_num = len(data_vars)

        vars = self.data.select_data(headers = data_vars)

        # for each subplot
        for i in range(var_num):
            for j in range(var_num):
                # get x and y data for each subplot
                x = vars[:,j]
                y = vars[:,i]
                # if the plot is not in the diagonal
                if i != j:
                    # calculate linear regression using spicy as default
                    self.linear_regression([data_vars[j]], data_vars[i], self.p)
                    # create x values for linear regression 
                    # (100 points equally spaced from x min to x max)
                    line_x = np.linspace(np.min(x), np.max(x), 100)
                    # create the linear regression with line_x
                    line_y = self.slope * line_x + self.intercept
                    line_y = line_y.flatten()
                    # label with resulting R^2
                    label = '$R^2$ = ' + str(round(self.R2, 3))
                    # plot linear regression in red 
                    axs[i,j].plot(line_x,line_y, c='red', label = label)
                    # write R^2 value in the title
                    axs[i,j].set_title(label)
                else: 
                    if hists_on_diag is True:
                        # makes the x and y axis scaling different for the histograms 
                        # (not shared with the scatter plots)
                        axs[i, j].remove()
                        axs[i, j] = fig.add_subplot(var_num, var_num, i*var_num+j+1)
                        if j < var_num-1:
                            axs[i, j].set_xticks([])
                        else:
                            axs[i, j].set_xlabel(data_vars[i])
                        if i > 0:
                            axs[i, j].set_yticks([])
                        else:
                            axs[i, j].set_ylabel(data_vars[i])
                        # create the historgram on the diagonal subplot
                        axs[i,j].hist(x)
        
        fig.suptitle("Linear Regression on each pair plot with $R^2$")
        fig.tight_layout()
                

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        N = A.shape[0]
        poly_matrix = np.ones((N, p))
        for i in range(p):
            poly_matrix[:,i] = np.squeeze(A**(i+1))
        return poly_matrix


    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and a column of homogeneous coordinates (1s).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        # select the variable columns associated with the independent and dependent variable strings
        ind_var = self.data.select_data(ind_var)
        self.A = self.make_polynomial_matrix(ind_var, p)
        self.y = self.data.select_data([dep_var])
        # set instance variable for polynomial regression degree
        self.p = p
        
        # Calculate the linear regression with the given method
        coeff = self.linear_regression_qr(self.A, self.y)

        # Last item in p (right most item) is the intercept
        # Remaning iterms are slopes
        self.slope = coeff[:len(coeff) - 1]
        self.slope = np.reshape(self.slope, (self.slope.shape[0], 1))
        self.intercept = coeff[len(coeff) - 1][0]

        # calculate the residuals
        y_pred = self.predict()
        res = self.compute_residuals(y_pred)
        self.residuals = res

        # calculate and print the R^2 value
        self.R2 = self.r_squared(y_pred)

        # set all instance variables in constructor
        self.ind_vars = ind_var
        self.dep_var = dep_var

        # calculate mean squared errors
        self.mse = self.compute_mse()

    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        # select the variable columns associated with the independent and dependent variable strings
        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data([dep_var])
        self.p = p 
        # set slope, intercept, and degree with the given values
        self.slope = slope
        self.intercept = intercept

        if self.p > 1:
            self.A = self.make_polynomial_matrix(self.A, self.p)
        
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred)
        self.mse = self.compute_mse()

    def pair_plot_extension(self, data_vars, fig_sz=(12, 12), hists_on_diag=True, p = 1):
        '''Makes a pair plot with polynomial regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.
        p: a int. polynomial degrees to plot the polynomial regression lines in each panel
        '''
        # Use pair_plot() in Analysis to take care of making the grid of scatter plots.
        fig, axs = analysis.Analysis.pair_plot(self, data_vars, fig_sz)

        # Create just a figure and only one subplot
        var_num = len(data_vars)

        vars = self.data.select_data(headers = data_vars)

        # for each subplot
        for i in range(var_num):
            for j in range(var_num):
                # get x and y data for each subplot
                x = vars[:,j]
                y = vars[:,i]
                # if the plot is not in the diagonal
                if i != j:
                    # calculate linear regression using spicy as default
                    self.poly_regression([data_vars[j]], data_vars[i], p)
                    # create x values for linear regression 
                    # (100 points equally spaced from x min to x max)
                    line_x = np.linspace(np.min(x), np.max(x), 100)

                    if p > 1:
                        line_x_s = self.make_polynomial_matrix(line_x, self.p) 

                    line_y = self.predict(line_x_s)

                    # label with resulting R^2
                    label = '$R^2$ = ' + str(round(self.R2, 3))
                    # plot linear regression in red 
                    axs[i,j].plot(line_x,line_y, c='red', label = label)
                    # write R^2 value in the title
                    axs[i,j].set_title(label)
                else: 
                    if hists_on_diag is True:
                        # makes the x and y axis scaling different for the histograms 
                        # (not shared with the scatter plots)
                        axs[i, j].remove()
                        axs[i, j] = fig.add_subplot(var_num, var_num, i*var_num+j+1)
                        if j < var_num-1:
                            axs[i, j].set_xticks([])
                        else:
                            axs[i, j].set_xlabel(data_vars[i])
                        if i > 0:
                            axs[i, j].set_yticks([])
                        else:
                            axs[i, j].set_ylabel(data_vars[i])
                        # create the historgram on the diagonal subplot
                        axs[i,j].hist(x)
        
        title = "Polynomial Regression of degree " + str(p) + " on each pair plot with $R^2$"
        fig.suptitle(title)
        fig.tight_layout()
        

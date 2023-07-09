'''analysis.py
Run statistical analyses and plot Numpy ndarray data
MICHELLE PHAN
CS 251 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

        # Make plot font sizes legible
        plt.rcParams.update({'font.size': 18})

    def set_data(self, data):
        '''Method that re-assigns the instance variable `data` with the parameter.
        Convenience method to change the data used in an analysis without having to create a new
        Analysis object.

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        self.data = data

    def min(self, headers, rows=[]):
        '''Computes the minimum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.
        (i.e. the minimum value in each of the selected columns)

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        # create a nparray with only the specific headers and rows
        subsets = self.data.select_data(headers, rows)
        # return the min along the axis 0 (along the column)
        return np.min(subsets, axis = 0)

    def max(self, headers, rows=[]):
        '''Computes the maximum of each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of max over, or over all indices
            if rows=[]

        Returns
        -----------
        maxs: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        # create a nparray with only the specific headers and rows
        subsets = self.data.select_data(headers, rows)
        # return the min along the axis 0 (along the column)
        return np.max(subsets, axis = 0)

    def range(self, headers, rows=[]):
        '''Computes the range [min, max] for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of min/max over, or over all indices
            if rows=[]

        Returns
        -----------
        mins: ndarray. shape=(len(headers),)
            Minimum values for each of the selected header variables
        maxes: ndarray. shape=(len(headers),)
            Maximum values for each of the selected header variables

        NOTE: There should be no loops in this method!
        '''
        min = self.min(headers, rows)
        max= self.max(headers, rows)
        return min,max

    def mean(self, headers, rows=[]):
        '''Computes the mean for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`).

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of mean over, or over all indices
            if rows=[]

        Returns
        -----------
        means: ndarray. shape=(len(headers),)
            Mean values for each of the selected header variables

        NOTE: You CANNOT use np.mean here!
        NOTE: There should be no loops in this method!
        '''
        # create a nparray with only the specific headers and rows
        subsets = self.data.select_data(headers, rows)
        # calculate sum of each headers and turn it into narray 
        sum = np.sum(subsets, axis = 0)
        sum = np.array(sum)
        # divide the sum by # of rows in the array 
        return sum / subsets.shape[0]

    def var(self, headers, rows=[]):
        '''Computes the variance for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of variance over, or over all indices
            if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Variance values for each of the selected header variables

        NOTE: You CANNOT use np.var or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        # create a nparray with only the specific headers and rows
        subsets = self.data.select_data(headers, rows)
        # find the mean 
        mean = self.mean(headers, rows)
        # Find each score’s deviation from the mean
        dev = subsets - mean
        # Square each deviation from the mean
        dev = np.square(dev)
        # Find the sum of squares
        dev = np.sum(dev,axis=0)
        # Divide the sum of squares by n – 1 (n = # of rows)
        dev = dev / (subsets.shape[0] - 1)
        return dev

    def std(self, headers, rows=[]):
        '''Computes the standard deviation for each variable in `headers` in the data object.
        Possibly only in a subset of data samples (`rows`) if `rows` is not empty.

        Parameters:
        -----------
        headers: Python list of str.
            One str per header variable name in data
        rows: Python list of int.
            Indices of data samples to restrict computation of standard deviation over,
            or over all indices if rows=[]

        Returns
        -----------
        vars: ndarray. shape=(len(headers),)
            Standard deviation values for each of the selected header variables

        NOTE: You CANNOT use np.var, np.std, or np.mean here!
        NOTE: There should be no loops in this method!
        '''
        return np.sqrt(self.var(headers, rows))

    def show(self):
        '''Simple wrapper function for matplotlib's show function.

        (Does not require modification)
        '''
        plt.show()

    def scatter(self, ind_var, dep_var, title=''):
        '''Creates a simple scatter plot with "x" variable in the dataset `ind_var` and
        "y" variable in the dataset `dep_var`. Both `ind_var` and `dep_var` should be strings
        in `self.headers`.

        Parameters:
        -----------
        ind_var: str.
            Name of variable that is plotted along the x axis
        dep_var: str.
            Name of variable that is plotted along the y axis
        title: str.
            Title of the scatter plot

        Returns:
        -----------
        x. ndarray. shape=(num_data_samps,)
            The x values that appear in the scatter plot
        y. ndarray. shape=(num_data_samps,)
            The y values that appear in the scatter plot

        NOTE: Do not call plt.show() here.
        '''
        # store data for x-axis
        x = self.data.select_data([ind_var])
        x = np.reshape(x, x.shape[0])

        # plotting a line plot after changing it's width and height
        f = plt.figure()
        f.set_figwidth(7)
        f.set_figheight(7)

        # label x axis
        plt.xlabel(ind_var)

        # store data for y-axis
        y = self.data.select_data([dep_var])
        y = np.reshape(y, y.shape[0])
        # label x axis
        plt.ylabel(dep_var)

        # plot the scatter graph
        plt.scatter(x,y)
        plt.title(title)

        f.tight_layout()
        
        return x,y

    def pair_plot(self, data_vars, fig_sz=(12, 12), title=''):
        '''Create a pair plot: grid of scatter plots showing all combinations of variables in
        `data_vars` in the x and y axes.

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        fig_sz: tuple of 2 ints.
            The width and height of the figure of subplots. Pass as a paramter to plt.subplots.
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        TODO:
        - Make the len(data_vars) x len(data_vars) grid of scatterplots
        - The y axis of the first column should be labeled with the appropriate variable being
        plotted there.
        - The x axis of the last row should be labeled with the appropriate variable being plotted
        there.
        - There should be no other axis or tick labels (it looks too cluttered otherwise!)

        Tip: Check out the sharex and sharey keyword arguments of plt.subplots.
        Because variables may have different ranges, pair plot columns usually share the same
        x axis and rows usually share the same y axis.
        '''
        # Create just a figure and only one subplot
        var_num = len(data_vars)
        fig, axs = plt.subplots(var_num, var_num, sharey='row', sharex='col')
        # set the firgue size according to fig_sz
        fig.set_figwidth(fig_sz[0])
        fig.set_figheight(fig_sz[1])

        vars = self.data.select_data(headers = data_vars)

        for i in range(var_num):
            for j in range(var_num):
                # only the x axis of last row be labeled
                if i == var_num - 1:
                    axs[i,j].set_xlabel(data_vars[j])

                # only the y axis of first row be labeled
                if j == 0:
                    axs[i,j].set_ylabel(data_vars[i])

                axs[i,j].scatter(vars[:,j], vars[:,i])

        fig.suptitle(title)
        fig.tight_layout()

        return fig, axs


    def boxplot(self, data_vars, title=''):
        '''Create a box plot: A box and whisker plot—also called a box 
        plot—displays the five-number summary of a set of data. 

        Parameters:
        -----------
        data_vars: Python list of str.
            Variables to place on either the x or y axis of the scatter plots
        title. str. Title for entire figure (not the individual subplots)

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        https://matplotlib.org/stable/plot_types/stats/boxplot_plot.html
        '''

        # Create just a figure and only one subplot
        vars = self.data.select_data(data_vars)
        
        # plot
        fig, ax = plt.subplots()
        # set the firgue size according to fig_sz
        fig.set_size_inches(12,12)

        VP = ax.boxplot(vars)

        fig.suptitle(title)
        fig.tight_layout()

        return fig, ax
    
    def bar(self, x, heights, title=''):
        '''Create a bar chart: Make a bar plot with subdata specified by data_vars

        Parameters:
        -----------
        x: Python str.
            Header of data for the x coordinates of the bars. 
        heights: Python str.
            Header of data for the height(s) of the bars.

        Returns:
        -----------
        fig. The matplotlib figure.
            1st item returned by plt.subplots
        axes. ndarray of AxesSubplot objects. shape=(len(data_vars), len(data_vars))
            2nd item returned by plt.subplots

        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
        '''
        # Create just a figure and only one subplot
        x_data = self.data.select_data([x])
        x_data = x_data.flatten()
        height_data = self.data.select_data([heights])
        height_data = height_data.flatten()

        # plot
        fig = plt.figure(figsize = (20, 5))
        plt.bar(x = x_data, height= height_data)
        plt.xlabel(x)
        plt.ylabel(heights)

        # set the firgue size according to fig_sz
        fig.set_size_inches(8,8)

        fig.suptitle(title)
        fig.tight_layout()


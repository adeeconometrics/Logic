try:
    import numpy as np
    import scipy as sci
    import scipy.special as ss
    # import sympy as sp
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))

# Current freatures: PDF, PMF, CDF generation with plotting (scatter plot, continous plotting)
'''
TODO:
- add a feature for generating numbers from a given distribution
- find a way to inteface with base class for plotting pdf and cdf 
'''


class Discrete:
    '''
    Base class for discrete probability distributions. This contains methods for initialization of values and plotting functions. 
    '''
    def __init__(self, data, points, xlim):
        self.data = data
        self.points = points
        self.xlim = xlim

    def scatter(self,
                func,
                xlim,
                ylim=None,
                ylabel=None,
                xlabel=None,
                threshold=100):
        '''
        Args: 
            func (function): pmf or cdf function 
            xlim(float): limit of x axis for plots
            ylim(float): limit of y axis for plots. Optional.
            ylabel(string): label of y axis. Optional.
            xlabel(string): label of x axis. Optional.
            threshold(int): points for plotting. Default value 100. 
        
        Returns: 
            plot of cdf or pmf 
        '''
        x = np.linspace(-xlim, xlim, threshold)
        y = func(x)
        if ylim != None:
            plt.ylim(-ylim, ylim)
        if ylabel != None:
            plt.ylabel(str(ylabel))
        if xlabel != None:
            plt.xlabel(str(xlabel))
        plt.scatter(x)


class Uniform(Discrete):
    def __init__(self, data, points, xlim):
        super().__init__(data, points, xlim)

    def pmf(self, scatter_plot=False):
        '''
        Args: 
            scatter_plot(bool): if true returns scatter plot. This is optional.

        Returns:
            probability mass value of uniform distribution or probability mass
            function's plot given the dataset.
        '''
        if scatter_plot == True:
            pass
        return 1 / len(self.data)

    def cdf(self, scatter_plot=False):
        '''
        Args: 
            scatter_plot(bool): if true returns scatter plot. This is optional.
            
        Returns:
            cumulative distribution value at a given point of uniform distribution or 
            cumuative distribution's plot given the dataset.
        '''
        if scatter_plot == True:
            pass
        pass


class binomial(Discrete):
    def pmf(self):
        pass

    def cdf(self):
        pass


class geometric(Discrete):
    def pmf(self):
        pass

    def cdf(self):
        pass


class poisson(Discrete):
    def pmf(self):
        pass

    def cdf(self):
        pass


class hypergeometric(Discrete):
    def pmf(self):
        pass

    def cdf(self):
        pass


class multinomial(Discrete):
    def pmf(self):
        pass

    def cdf(self):
        pass


class Continuous:
    '''
    This class contains methods for continuous probability distirbutions and their corresponding PDF and CDF
    '''
    def __init__(self, data):
        self.data = data

    def plot(self,
             func,
             xlim,
             ylim=None,
             ylabel=None,
             xlabel=None,
             threshold=1000):
        '''
        Args:
            func(function):
            xlim(float)
            ylim(float)
            ylabel(string)
            xlabel(string)
            threshold(int)
        '''
        x = np.linspace(-xlim, xlim, threshold)
        y = func(x)
        if ylim != None:
            plt.ylim(-ylim, ylim)
        if ylabel != None:
            plt.ylabel(str(ylabel))
        if xlabel != None:
            plt.xlabel(str(xlabel))
        plt.plot(x, y)


class normal(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class t_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class cauchy_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class f_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class chi_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class gamma_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class pareto_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class log_normal_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass


class non_central_chi_dist(Continuous):
    def pdf(self):
        pass

    def cdf(self):
        pass
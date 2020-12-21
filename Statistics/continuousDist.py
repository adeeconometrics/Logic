try:
    import numpy as np
    import scipy as sci
    import scipy.special as ss
    # import sympy as sp
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))


class Base:
    def __init__(self, data):
        self.data = data

    # add fill-color function given some condition
    def plot(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
        if ylim is not None:
            plt.ylim(0, ylim)  # scales from 0 to ylim
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.plot(x, y)


# check for quality assessment
class Uniform:
    '''
    This contains methods for finding the probability density function and 
    cumulative distirbution function of Uniform distribution. Incudes plotting method. 

    Args: 
        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

    Methods
        - pmf for probability density function
        - cdf for cumulative distribution function
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        '''
        Args:
            plot (bool): returns scatter plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        '''
        a = self.a
        b = self.b
        threshold = b - a

        generator = lambda a, b, x: 1 / (b - a) if a <= x and x <= b else 0
        if plot == True:
            x = np.linspace(a, b, threshold)
            y = np.array([generator(a, b, i) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(a, b, np.abs(b - a))

    def cdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        '''
        Args:
            plot (bool): returns scatter plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        '''
        a = self.a
        b = self.b
        threshold = b - a

        def generator(a, b, x):
            if x < a:
                return 0
            if (a <= x and x <= b):
                return (x - a) / (b - a)
            if x > b:
                return 1

        if plot == True:
            x = np.linspace(a, b, threshold)
            y = np.array([generator(a, b, i) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(a, b, threshold)  # what does it really say?


# add method for computing normal distribution given (self.mean, self.std, self.randvar)
class Normal(Base):
    '''
    This class contains implementation of the Normal Distribution for calculating the
    probablity density function and cumulative distirbution function. Additionally, 
    a z-table generator is also provided by p-value method.
    Args: 
        mean(float): mean of the distribution
        std(float): standard deviation of the distribution
        randvar(float∈[0,1]): random variable

    Methods:
        pdf - returns either plot of the distribution or evaluation at randvar.
        cdf - returns either plot of the distirbution or evaluation at randvar.
        p_value - returns p-value at randvar.
    '''
    def __init__(self, mean, std, x):
        self.mean = mean
        self.std = std
        self.randvar = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 
        
        Returns:
            either plot of the distribution or probability density evaluation at randvar.
        '''
        mean = self.mean
        std = self.std
        generator = lambda x, mean, std: np.power(
            1 / (std * np.sqrt(2 * np.pi)), np.exp(((x - mean) / 2 * std)**2))
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([generator(x_temp, mean, std) for x_temp in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(self.randvar, mean, std)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):  # cdf ~ z-score?
        '''
        Args: 
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either plot of the distirbution or cumulative density evaluation at randvar.
        '''
        randvar = self.randvar
        generator = lambda x: ss.erf(x)
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([generator(x_temp) for x_temp in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(randvar)

    def p_value(self, x_lower=-(np.inf), x_upper=None):
        '''
        Args:
            x_lower(float): defaults to infinity.
            x_upper(float): defaults to None. 

        Returns:
            p-value drawn from normal distirbution between x_lower and x_upper.
        '''
        generator = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)
        return sci.integrate.quad(generator, x_lower, x_upper)[0]


class T_distribution(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Cauchy(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class F_distribution(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Chi_distribution(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Gamma_distribution(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Pareto(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Log_normal(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass


class Non_central_chi(Base):
    def __init__(self):
        pass

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):

        pass

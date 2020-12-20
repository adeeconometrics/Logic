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


class Base:
    def __init__(self, data):
        self.data = data

    def scatter(self, x, y, xlim=None, xlabel=None, ylabel=None):
        if xlim is not None:
            plt.xlim(-xlim, xlim)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.scatter(x, y)


class Uniform(Base):
    '''
    This contains methods for finding the probability mass function and 
    cumulative distirbution function of Uniform distribution. Incudes scatter plot. 

    Args: 
        data (int): sample size
    Methods
        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, data):
        self.data = np.ones(data)

    def pmf(self, plot=False):
        '''
        Args:
            plot (bool): returns scatter plot if true. 

        Returns:
            either probability mass value of Uniform distirbution or scatter plot
        '''
        if plot == True:
            x = np.array([i for i in range(0, len(self.data))])
            y = np.array(
                [1 / len(self.data) for i in range(0, len(self.data))])
            super().scatter(x, y)
        return 1 / len(self.data)

    def cdf(self, a, b, point=0, plot=False):
        '''
        Args:
            a(int): lower limit of the distirbution
            b(int): upper limit of the distribution
            point(int): point at which cumulative value is evaluated. Optional. 
            plot(bool): returns plot if true.

        Retruns:
            either cumulative distribution evaluation at some point or scatter plot.
        '''

        cdf_function = lambda x, _a, _b: (np.floor(x) - _a + 1) / (_b - _a + 1)
        if plot == True:
            x = np.array([i + 1 for i in range(a, b)])
            y = np.array([cdf_function(i, a, b) for i in x])
            super().scatter(x, y)
        return cdf_function(point, a, b)


# resolve issue on CDF
class Binomial(Base):
    '''
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for binomial distirbution. 

    Args:
        data(int): sample points for the scatterplot
        n(int): number  of trials 
        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 
        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, data, n, p, k):
        super().__init__(data)
        self.n = n
        self.p = p
        self.k = k

    def pmf(self, interval=None, threshold=100, plot=False):
        '''
        Args:
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
        
        Returns: 
            either probability mass evaluation for some point or scatter plot of binomial distribution.
        '''
        n = self.n
        p = self.p
        k = self.k

        def generator(n, p, k):
            bin_coef = lambda n, k: ss.binom(n, k)  # assumes n>k
            if isinstance(k, list) == True:
                k_list = [i + 1 for i in range(0, len(k))]
                y = np.array([(bin_coef(n, k_) * np.power(p, k_)) *
                              np.power(1 - p, n - k_) for k_ in k_list])
                return y
            return (bin_coef(n, k) * np.power(p, k)) * np.power(1 - p, n - k)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = generator(n, p, x)
            super().scatter(x, y)

        return generator(n, p, k)

    def cdf(self, interval=0, point=0, threshold=100, plot=False):
        '''
        Args:
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
        
        Returns: 
            either cumulative distirbution evaluation for some point or scatter plot of binomial distribution.
        '''
        n = self.n
        p = self.p
        k = self.k

        def generator(n, p, k):
            bin_coef = lambda x: np.array(
                (np.math.factorial(n) /
                 (np.math.factorial(x) * np.math.factorial(np.abs(n - x)))) *
                (np.power(p, x) * np.power((1 - p), n - x)))
            return np.cumsum([bin_coef(j) for j in range(0, k)], dtype=float)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = generator(n, p, len(x))
            super().scatter(x, y)

        return generator(n, p, point)[
            point -
            1]  # will this output the cumulative sum at point requested?


class Multinomial(Base):
    '''
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for mutlinomial distirbution. 

    Args:
        data(int): sample points for the scatterplot
        n(int): number  of trials 
        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 
        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, data):
        super().__init__(data)

    def pmf(self):
        pass

    def cdf(self):
        pass


class Geometric(Base):
    '''
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for geometric distirbution. We consider two definitions 
    of the geometric distribution: one concerns itself to the number of X of Bernoulli trials
    needed to get one success, supported on the set {1,2,3,...}. The second one concerns with 
    Y=X-1 of failures before the first success, supported on the set {0,1,2,3,...}. 

    Args:
        data(int): sample points for the scatterplot
        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 
        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, data, p, k):
        super().__init__(data)
        self.p = p
        self.k = k

    def pmf(self, interval=None, threshold=100, plot=False, type=first):
        '''
        Args: 
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.           
            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            Reference: https://en.wikipedia.org/wiki/Geometric_distribution

        Returns: 
            probability mass evaluation to some point specified by k or scatter plot of geometric distribution.
            
        Note: there are two configurations of pmf. 
        '''
        p = self.p
        k = self.k
        if type == "first":
            generator = lambda p, k: np.power(1 - p, k - 1) * p
        elif type == "second":
            generator = lambda p, k: np.power(1 - p, k) * p
        else:  # supposed to raise exception when failed
            print("Invalid argument. Type is either 'first' or 'second'.")

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(p, k_i) for k_i in x])
            super().scatter(x, y)

        return generator(p, k)

    def cdf(self, interval=None, threshold=100, plot=False, type=first):
        '''
        Args: 
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.           
            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            for context see: https://en.wikipedia.org/wiki/Geometric_distribution

        Returns: 
            cumulative distirbution evaluation to some point specified by k or scatter plot of geometric distribution.
            
        Note: there are two configurations of cdf. 
        '''
        p = self.p
        k = self.k
        if type == "first":
            generator = lambda p, k: 1 - np.power(1 - p, k)
        elif type == "second":
            generator = lambda p, k: 1 - np.power(1 - p, k + 1)
        else:  # supposed to raise exception when failed
            print("Invalid argument. Type is either 'first' or 'second'.")

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(p, k_i) for k_i in x])
            super().scatter(x, y)

        return generator(p, k)


class Hypergeometric(Base):
    '''
    Args:
        N(int): population size
        K(int): number of success states in the population
        k(int): number of observed successes
        n(int): number of draws 
    '''
    def __init__(self, data, N, K, k, n):
        super().__init__(data)
        self.N = N
        self.K = K
        self.k = k
        self.n = n

    def pmf(self, interval=None, threshold=100, plot=False):
        '''
        Args:
            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.
        
        Returns: 
            either probability mass evaluation for some point or scatter plot of hypergeometric distribution.
        '''
        n = self.n
        k = self.k
        N = self.N
        K = self.K

        generator = lambda N, n, K, k: (ss.binom(n, k) * ss.binom(
            N - K, n - k)) / ss.binom(N, n)  # assumes n>k

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array(
                [generator(N, n, K, x_temp) for x_temp in range(0, len(x))])
            super().scatter(x, y)

        return generator(N, n, K, k)

    def cdf(self):  # np.cumsum()
        pass


class Poisson(Base):
    def __init__(self, data):
        super().__init__(data)

    def pmf(self):
        pass

    def cdf(self):
        pass
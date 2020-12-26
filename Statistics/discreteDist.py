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
'''


class Base:  # add histograms
    def __init__(self, data):
        self.data = data

    def scatter(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
        if ylim is not None:
            plt.ylim(0, ylim)  # scales from 0 to ylim
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

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
    '''
    def __init__(self, data):
        self.data = np.ones(data)

    def pmf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        '''
        Args:

            plot (bool): returns scatter plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either probability mass value of Uniform distirbution or scatter plot
        '''
        if plot == True:
            x = np.array([i for i in range(0, len(self.data))])
            y = np.array(
                [1 / len(self.data) for i in range(0, len(self.data))])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return 1 / len(self.data)

    def cdf(self,
            a,
            b,
            point=0,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:

            a(int): lower limit of the distirbution
            b(int): upper limit of the distribution
            point(int): point at which cumulative value is evaluated. Optional. 
            plot(bool): returns plot if true.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 


        Retruns:
            either cumulative distribution evaluation at some point or scatter plot.
        '''

        cdf_function = lambda x, _a, _b: (np.floor(x) - _a + 1) / (_b - _a + 1)
        if plot == True:
            x = np.array([i + 1 for i in range(a, b)])
            y = np.array([cdf_function(i, a, b) for i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return cdf_function(point, a, b)


# resolve issue on CDF
class Binomial(Base):
    '''
    This class contains functions for finding the probability mass function and 
    cumulative distribution function for binomial distirbution. 

    Args:

        n(int): number  of trials 
        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
    '''
    def __init__(self, n, p, k):
        self.n = n
        self.p = p
        self.k = k

    def pmf(self,
            interval=None,
            threshold=100,
            plot=False,
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
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(n, p, k)

    def cdf(self,
            interval=0,
            point=0,
            threshold=100,
            plot=False,
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
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

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

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
    '''
    def __init__(self, data):
        super(Multinomial, self).__init__(data)

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

        p(float ∈ [0,1]): success probability for each trial
        k(int): number of successes 

    Methods: 

        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, p, k):
        self.p = p
        self.k = k

    def pmf(self,
            interval=None,
            threshold=100,
            plot=False,
            type=first,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args: 

            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.           
            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

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
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(p, k)

    def cdf(self,
            interval=None,
            threshold=100,
            plot=False,
            type=first,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args: 

            interval(int): defaults to none. Only necessary for defining scatter plot.
            threshold(int): defaults to 100. Defines the sample points in scatter plot.
            plot(bool): if true, returns scatter plot.           
            type(keyvalue ∈[fist, second]): defaults to first. Reconfigures the type of distribution.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

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
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(p, k)


class Hypergeometric(Base):
    '''
    This class contains methods concerning pmf and cdf evaluation of the hypergeometric distribution. 
    Describes the probability if k successes (random draws for which the objsect drawn has specified deature)
    in n draws, without replacement, from a finite population size N that contains exactly K objects with that
    feature, wherein each draw is either a success or a failure. 

    Args:

        N(int): population size
        K(int): number of success states in the population
        k(int): number of observed successes
        n(int): number of draws 
    
    Methods: 

        - pmf for probability mass function
        - cdf for cumulative distribution function
    '''
    def __init__(self, N, K, k, n):
        self.N = N
        self.K = K
        self.k = k
        self.n = n

    def pmf(self,
            interval=None,
            threshold=100,
            plot=False,
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

            Reference: https://en.wikipedia.org/wiki/Hypergeometric_distribution
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
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(N, n, K, k)

    def cdf(self):  # np.cumsum()
        pass


class Poisson(Base):
    '''
    This class contains methods for evaluating some properties of the poisson distribution. 
    As lambda increases to sufficiently large values, the normal distribution (λ, λ) may be used to 
    approximate the Poisson distribution.

    Use the Poisson distribution to describe the number of times an event occurs in a finite observation space.
    
    References: Minitab (2019). Poisson Distribution. https://bityl.co/4uYc

    Args: 

        λ(float): expected rate if occurrences.
        k(int): number of occurrences.
    '''
    def __init__(self, λ, k):
        self.k = k
        self.λ = λ

    def pmf(self,
            interval=None,
            threshold=100,
            plot=False,
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

            Reference: https://en.wikipedia.org/wiki/Poisson_distribution
        
        Returns: 
            either probability mass evaluation for some point or scatter plot of poisson distribution.
        '''
        k = self.k
        λ = self.λ
        generator = lambda k, λ: (np.power(λ, k) * np.exp(-λ)
                                  ) / np.math.factorial(k)
        if plot == True:
            x = np.linspace(1, interval, threshold)
            y = np.array([generator(x_temp, λ) for x_temp in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return generator(k, λ)

    def cdf(self,
            interval=None,
            threshold=100,
            plot=False,
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

            Reference: https://en.wikipedia.org/wiki/Poisson_distribution
        
        Returns: 
            either cumulative distribution evaluation for some point or scatter plot of poisson distribution.
        '''
        k = self.k
        λ = self.λ
        generator = lambda k, λ: ss.gammaic(np.floor(k + 1), λ
                                            ) / np.math.factorial(np.floor(k))
        if plot == True:
            x = np.linspace(1, interval, threshold)
            y = np.array([generator(x_temp, λ) for x_temp in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return generator(k, λ)
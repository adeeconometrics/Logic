try:
    import numpy as np
    import scipy as sci
    import scipy.special as ss
    # import sympy as sp
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))
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

    # def hist(self, x, y, xlim=None, ylim=None, xlabel=None, ylabel=None):
    #     if ylim is not None:
    #         plt.ylim(0, ylim)  # scales from 0 to ylim
    #     if xlim is not None:
    #         plt.xlim(-xlim, xlim)
    #     if xlabel is not None:
    #         plt.xlabel(xlabel)
    #     if ylabel is not None:
    #         plt.ylabel(ylabel)
    #     plt.scatter(x, y)


class Uniform(Base):
    '''
    This contains methods for finding the probability mass function and 
    cumulative distirbution function of Uniform distribution. Incudes scatter plot. 

    Args: 

        data (int): sample size

    Methods

        - pdf for evaluating or plotting probability mass function
        - cdf for evaluating or plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    Reference:
    - NIST/SEMATECH e-Handbook of Statistical Methods (2012). Uniform Distribution. Retrieved from http://www.itl.nist.gov/div898/handbook/, December 26, 2020.
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

    def mean(self):
        '''
        Returns the mean of Uniform Distribution.
        '''
        return (self.a + self.b) / 2

    def median(self):
        '''
        Returns the median of Uniform Distribution.
        '''
        return (self.a + self.b) / 2

    def mode(self):
        '''
        Returns the mode of Uniform Distribution.
        '''
        return (self.a, self.b)

    def var(self):
        '''
        Returns the variance of Uniform Distribution.
        '''
        return (self.b - self.a)**2 / 12

    def skewness(self):
        '''
        Returns the skewness of Uniform Distribution.
        '''
        return 0

    def kurtosis(self):
        '''
        Returns the kurtosis of Uniform Distribution.
        '''
        return -6 / 5

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


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
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
    - NIST/SEMATECH e-Handbook of Statistical Methods (2012). Binomial Distribution. 
    Retrieved at http://www.itl.nist.gov/div898/handbook/, December 26, 2000.
    - Wikipedia contributors. (2020, December 19). Binomial distribution. 
    In Wikipedia, The Free Encyclopedia. Retrieved 07:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Binomial_distribution&oldid=995095096
    - Weisstein, Eric W. "Binomial Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/BinomialDistribution.html
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

    def mean(self):
        '''
        Returns the mean of Binomial Distribution.
        '''
        return self.n * self.p

    def median(self):
        '''
        Returns the median of Binomial Distribution. Either one defined in the tuple of result.
        '''
        n = self.n
        p = self.p
        return np.floor(n * p), np.ciel(n * p)

    def mode(self):
        '''
        Returns the mode of Binomial Distribution. Either one defined in the tuple of result.
        '''
        n = self.n
        p = self.p
        return np.floor((n + 1) * p), np.ceil((n + 1) * p) - 1

    def var(self):
        '''
        Returns the variance of Binomial Distribution.
        '''
        n = self.n
        p = self.p
        q = 1 - p
        return n * p * q

    def skewness(self):
        '''
        Returns the skewness of Binomial Distribution.
        '''
        n = self.n
        p = self.p
        q = 1 - p
        return (q - p) / np.sqrt(n * p * q)

    def kurtosis(self):
        '''
        Returns the kurtosis of Binomial Distribution.
        '''
        n = self.n
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (n * p * q)

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


# class Multinomial(Base):
#     '''
#     This class contains functions for finding the probability mass function and
#     cumulative distribution function for mutlinomial distirbution.

#     Args:

#         data(int): sample points for the scatterplot
#         n(int): number  of trials
#         p(float ∈ [0,1]): success probability for each trial
#         k(int): number of successes

#     Methods:

#         - pmf for probability mass function.
#         - cdf for cumulative distribution function.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - print_summary for printing summary statistics.
#     '''
#     def __init__(self, data):
#         super(Multinomial, self).__init__(data)

#     def pmf(self):
#         pass

#     def cdf(self):
#         pass

#     def mean(self):
#         '''
#         Returns the mean of Multinomial Distribution.
#         '''
#         pass

#     def median(self):
#         '''
#         Returns the median of Multinomial Distribution.
#         '''
#         pass

#     def mode(self):
#         '''
#         Returns the mode of Multinomial Distribution.
#         '''
#         pass

#     def var(self):
#         '''
#         Returns the variance of Multinomial Distribution.
#         '''
#         pass

#     def skewness(self):
#         '''
#         Returns the skewness of Multinomial Distribution.
#         '''
#         pass

#     def kurtosis(self):
#         '''
#         Returns the kurtosis of Multinomial Distribution.
#         '''
#         pass

#     def print_summary(self):
#         '''
#         Returns: Summary statistic regarding the distribution
#         '''
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = "summary statistic"
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


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

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
    - Weisstein, Eric W. "Geometric Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/GeometricDistribution.html
    - Wikipedia contributors. (2020, December 27). Geometric distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:05, December 27, 2020, from https://en.wikipedia.org/w/index.php?title=Geometric_distribution&oldid=996517676
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
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(p, k_i) for k_i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(p, k)

    def mean(self, type=first):
        '''
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the mean of Geometric Distribution.
        '''
        p = self.p
        if type == "first":
            return 1 / p
        elif type == "second":
            return (1 - p) / p
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def median(self, type=first):
        '''
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the median of Geometric Distribution.
        '''
        if type == "first":
            return np.ciel(1 / (np.log2(1 - self.p)))
        elif type == "second":
            return np.ciel(1 / (np.log2(1 - self.p))) - 1
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def mode(self, type=first):
        '''
        Args:

            type(string): defaults to first type. Valid types: "first", "second".
        Returns the mode of Geometric Distribution.
        '''
        if type == "first":
            return 1
        elif type == "second":
            return 0
        else:  # supposed to raise exception when failed
            return print(
                "Invalid argument. Type is either 'first' or 'second'.")

    def var(self):
        '''
        Returns the variance of Geometric Distribution.
        '''
        return (1 - self.p) / self.p**2

    def skewness(self):
        '''
        Returns the skewness of Geometric Distribution.
        '''
        return (2 - self.p) / np.sqrt(1 - self.p)

    def kurtosis(self):
        '''
        Returns the kurtosis of Geometric Distribution.
        '''
        return 6 + (self.p**2 / (1 - self.p))

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


# possible issue needs to be fixed with parameters
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

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
    - Weisstein, Eric W. "Hypergeometric Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/HypergeometricDistribution.html
    - Wikipedia contributors. (2020, December 22). Hypergeometric distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 08:38, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Hypergeometric_distribution&oldid=995715954

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

    def cdf(self,
            interval=None,
            threshold=100,
            plot=False,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):  # np.cumsum()
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
            either cumulative distribution evaluation for some point or scatter plot of hypergeometric distribution.
        '''
        n = self.n
        k = self.k
        N = self.N
        K = self.K

        generator = lambda N, n, K, k: (ss.binom(n, k) * ss.binom(
            N - K, n - k)) / ss.binom(N, n)  # assumes n>k

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.cumsum(
                [generator(N, n, K, x_temp) for x_temp in range(0, len(x))])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)
        return np.cumsum(generator(N, n, K, k))[k - 1]

    def mean(self):
        '''
        Returns the mean of Hypergeometric Distribution.
        '''
        return self.n * (self.K / self.N)

    def median(self):
        '''
        Returns the median of Hypergeometric Distribution. Currently unsupported or undefined.
        '''
        return "undefined"

    def mode(self):
        '''
        Returns the mode of Hypergeometric Distribution.
        '''
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return np.ceil(((n + 1) * (K + 1)) / (N + 2)) - 1, np.floor(
            ((n + 1) * (K + 1)) / (N + 2))

    def var(self):
        '''
        Returns the variance of Hypergeometric Distribution.
        '''
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))

    def skewness(self):
        '''
        Returns the skewness of Hypergeometric Distribution.
        '''
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        return ((N - 2 * K) * np.power(N - 1, 1 / 2) *
                (N - 2 * n)) / (np.sqrt(n * K * (N - K) * (N - n)) * (N - 2))

    def kurtosis(self):
        '''
        Returns the kurtosis of Hypergeometric Distribution.
        '''
        n = self.n
        N = self.N
        k = self.k
        K = self.K
        scale = 1 / (n * k(N - K) * (N - n) * (N - 2) * (N - 3))
        return scale * ((N - 1) * N**2 * (N * (N + 1) - (6 * K * (N - K)) -
                                          (6 * n * (N - n))) +
                        (6 * n * K(N - K) * (N - n) * (5 * N - 6)))

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


class Poisson(Base):
    '''
    This class contains methods for evaluating some properties of the poisson distribution. 
    As lambda increases to sufficiently large values, the normal distribution (λ, λ) may be used to 
    approximate the Poisson distribution.

    Use the Poisson distribution to describe the number of times an event occurs in a finite observation space.
    

    Args: 

        λ(float): expected rate if occurrences.
        k(int): number of occurrences.

    Methods:

        - pmf for probability mass function.
        - cdf for cumulative distribution function.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
        -  Minitab (2019). Poisson Distribution. https://bityl.co/4uYc
        - Weisstein, Eric W. "Poisson Distribution." From MathWorld--A Wolfram Web Resource. 
        https://mathworld.wolfram.com/PoissonDistribution.html
        - Wikipedia contributors. (2020, December 16). Poisson distribution. In Wikipedia, The Free Encyclopedia.
         Retrieved 08:53, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Poisson_distribution&oldid=994605766
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

    def mean(self):
        '''
        Returns the mean of Poisson Distribution.
        '''
        return self.λ

    def median(self):
        '''
        Returns the median of Poisson Distribution.
        '''
        λ = self.λ
        return λ + 1 / 3 - (0.02 / λ)

    def mode(self):
        '''
        Returns the mode of Poisson Distribution.
        '''
        λ = self.λ
        return np.ceil(λ) - 1, np.floor(λ)

    def var(self):
        '''
        Returns the variance of Poisson Distribution.
        '''
        return self.λ

    def skewness(self):
        '''
        Returns the skewness of Poisson Distribution.
        '''
        return np.power(self.λ, -1 / 2)

    def kurtosis(self):
        '''
        Returns the kurtosis of Poisson Distribution.
        '''
        return np.power(self.λ, -1)

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


# class Beta_binomial(Base):
#     '''
#     Args:

#     Methods:

#         - pmf for evaluating or plotting probability mass function
#         - cdf for evaluating or plotting cumulative distribution function
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - print_summary for printing summary statistics.

#     '''
#     def __init__(self):
#         pass

#     def pmf(self,
#             interval=None,
#             threshold=100,
#             plot=False,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:

#             interval(int): defaults to none. Only necessary for defining scatter plot.
#             threshold(int): defaults to 100. Defines the sample points in scatter plot.
#             plot(bool): if true, returns scatter plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.

#         Returns:
#             either probability mass evaluation for some point or scatter plot of Beta Binomial distribution.
#         '''
#         pass

#     def cdf(self,
#             interval=None,
#             threshold=100,
#             plot=False,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:

#             interval(int): defaults to none. Only necessary for defining scatter plot.
#             threshold(int): defaults to 100. Defines the sample points in scatter plot.
#             plot(bool): if true, returns scatter plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.

#         Returns:
#             either cumulative distribution evaluation for some point or scatter plot of Beta Binomial distribution.
#         '''
#         pass

#     def mean(self):
#         '''
#         Returns the mean of Beta Binomial Distribution.
#         '''
#         pass

#     def median(self):
#         '''
#         Returns the median of Beta Binomial Distribution.
#         '''
#         pass

#     def mode(self):
#         '''
#         Returns the mode of Beta Binomial Distribution.
#         '''
#         pass

#     def var(self):
#         '''
#         Returns the variance of Beta Binomial Distribution.
#         '''
#         pass

#     def skewness(self):
#         '''
#         Returns the skewness of Beta Binomial Distribution.
#         '''
#         pass

#     def kurtosis(self):
#         '''
#         Returns the kurtosis of Beta Binomial Distribution.
#         '''
#         pass

#     def print_summary(self):
#         '''
#         Returns: Summary statistic regarding the distribution
#         '''
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = "summary statistic"
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Bernoulli(Base):
    '''
    This class contains methods concerning the Bernoulli Distribution. Bernoulli Distirbution is a special
    case of Binomial Distirbution. 
    Args:

        - p(int): event of success. 
        - k(float ∈[0,1]): possible outcomes
    Methods:

        - pmf for evaluating or plotting probability mass function
        - cdf for evaluating or plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
        - Weisstein, Eric W. "Bernoulli Distribution." From MathWorld--A Wolfram Web Resource. 
        https://mathworld.wolfram.com/BernoulliDistribution.html
        - Wikipedia contributors. (2020, December 26). Bernoulli distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:18, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Bernoulli_distribution&oldid=996380822
    '''
    def __init__(self, p, k):
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
            either probability mass evaluation for some point or scatter plot of Bernoulli distribution.
        '''
        p = self.p
        k = self.k
        generator = lambda p, k: p**k * np.power(1 - p, 1 - k)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(p, i) for i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(p, k)

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

        
        Returns: 
            either cumulative distribution evaluation for some point or scatter plot of Bernoulli distribution.
        '''
        p = self.p
        k = self.k

        def generator(k, p):
            if k < 0:
                return 0
            elif k >= 0 and k < 1:
                return 1 - p
            elif k >= 1:
                return 1

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(p, i) for i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(p, k)

    def mean(self):
        '''
        Returns the mean of Bernoulli Distribution.
        '''
        return self.p

    def median(self):
        '''
        Returns the median of Bernoulli Distribution.
        '''
        p = self.p
        if p < 1 / 2:
            return 0
        if p == 1 / 2:
            return [0, 1]
        if p > 1 / 2:
            return 1

    def mode(self):
        '''
        Returns the mode of Bernoulli Distribution.
        '''
        p = self.p
        if p < 1 / 2:
            return 0
        if p == 1 / 2:
            return 0, 1
        if p > 1 / 2:
            return 1

    def var(self):
        '''
        Returns the variance of Bernoulli Distribution.
        '''
        p = self.p
        q = 1 - p
        return p * q

    def skewness(self):
        '''
        Returns the skewness of Bernoulli Distribution.
        '''
        p = self.p
        q = 1 - p
        return (q - p) / np.sqrt(p * q)

    def kurtosis(self):
        '''
        Returns the kurtosis of Bernoulli Distribution.
        '''
        p = self.p
        q = 1 - p
        return (1 - 6 * p * q) / (p * q)

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


# class Negative_binomial(Base):
#     '''
#     This class contains methods concerning the Negative Binomial Distribution.

#     Args:

#     Methods:

#         - pmf for evaluating or plotting probability mass function
#         - cdf for evaluating or plotting cumulative distribution function
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - print_summary for printing summary statistics.

#     '''
#     def __init__(self):
#         pass

#     def pmf(self,
#             interval=None,
#             threshold=100,
#             plot=False,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:

#             interval(int): defaults to none. Only necessary for defining scatter plot.
#             threshold(int): defaults to 100. Defines the sample points in scatter plot.
#             plot(bool): if true, returns scatter plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.

#         Returns:
#             either probability mass evaluation for some point or scatter plot of Negative Binomial distribution.
#         '''
#         pass

#     def cdf(self,
#             interval=None,
#             threshold=100,
#             plot=False,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:

#             interval(int): defaults to none. Only necessary for defining scatter plot.
#             threshold(int): defaults to 100. Defines the sample points in scatter plot.
#             plot(bool): if true, returns scatter plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true.
#             xlabel(string): sets label in x axis. Only relevant when plot is true.
#             ylabel(string): sets label in y axis. Only relevant when plot is true.

#         Returns:
#             either cumulative distribution evaluation for some point or scatter plot of Negative Binomial distribution.
#         '''
#         pass

#     def mean(self):
#         '''
#         Returns the mean of Negative Binomial Distribution.
#         '''
#         pass

#     def median(self):
#         '''
#         Returns the median of Negative Binomial Distribution.
#         '''
#         pass

#     def mode(self):
#         '''
#         Returns the mode of Negative Binomial Distribution.
#         '''
#         pass

#     def var(self):
#         '''
#         Returns the variance of Negative Binomial Distribution.
#         '''
#         pass

#     def skewness(self):
#         '''
#         Returns the skewness ofNegative Binomial Distribution.
#         '''
#         pass

#     def kurtosis(self):
#         '''
#         Returns the kurtosis of Negative Binomial Distribution.
#         '''
#         pass
#     def print_summary(self):
#         '''
#         Returns: Summary statistic regarding the distribution
#         '''
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = "summary statistic"
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# issue: cdf
class Zeta(Base):
    '''
    This class contains methods concerning the Zeta Distribution.

    Args:
        - s(float): main parameter
        - k(int): support parameter
    Methods:

        - pmf for evaluating or plotting probability mass function
        - cdf for evaluating or plotting cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing summary statistics.

    References:
        - Wikipedia contributors. (2020, November 6). Zeta distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:24, December 26, 2020, from https://en.wikipedia.org/w/index.php?title=Zeta_distribution&oldid=987351423
    '''
    def __init__(self, s, k):
        s = self.s
        k = self.k

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
            either probability mass evaluation for some point or scatter plot of Zeta distribution.
        '''
        s = self.s
        k = self.k
        generator = lambda s, k: (1 / k**6) / ss.zeta(s)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(s, i) for i in x])
            return super().scatter(x, y, xlim, ylim, xlabel, ylabel)

        return generator(s, k)

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

        
        Returns: 
            either cumulative distribution evaluation for some point or scatter plot of Zeta distribution.
        '''
        pass

    def mean(self):
        '''
        Returns the mean of Zeta Distribution. Returns None if undefined.
        '''
        s = self.s
        if s > 2:
            return ss.zeta(s - 1) / ss.zeta(s)
        return "undefined"

    def median(self):
        '''
        Returns the median of Zeta Distribution. Retruns None if undefined.
        '''
        return "undefined"

    def mode(self):
        '''
        Returns the mode of Zeta Distribution.
        '''
        return 1

    def var(self):
        '''
        Returns the variance of Zeta Distribution. Returns None if undefined.
        '''
        s = self.s
        if s > 3:
            return (ss.zeta(s) * ss.zeta(s - 1) -
                    ss.zeta(s - 1)**2) / ss / zeta(s)**2
        return "undefined"

    def skewness(self):
        '''
        Returns the skewness of Zeta Distribution. Currently unsupported.
        '''
        return "unsupported"

    def kurtosis(self):
        '''
        Returns the kurtosis of Zeta Distribution. Currently unsupported.
        '''
        return "unsupported"

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)

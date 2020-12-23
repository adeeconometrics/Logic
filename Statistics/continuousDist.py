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
    This class contains methods concerning the Continuous Uniform Distribution.

    Args: 

        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

    Methods

        - pmf for probability density function
        - cdf for cumulative distribution function

    Referene:
    - Weisstein, Eric W. "Uniform Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/UniformDistribution.html
    '''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        '''
        Args:

            plot (bool): returns plot if true. 
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

            plot (bool): returns plot if true. 
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


# add method for confidence intervals
class Normal(Base):
    '''
    This class contains methods concerning the Standard Normal Distribution.

    Args: 

        mean(float): mean of the distribution
        std(float): standard deviation of the distribution
        randvar(float∈[0,1]): random variable

    Methods:

        pdf - returns either plot of the distribution or evaluation at randvar.
        cdf - returns either plot of the distirbution or evaluation at randvar.
        p_value - returns p-value at randvar.

    References:
    - Wikipedia contributors. (2020, December 19). Normal distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:44, 
    December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Normal_distribution&oldid=995237372
    - Weisstein, Eric W. "Normal Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/NormalDistribution.html

    '''
    def __init__(self, x, mean=0, std=1):
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

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
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

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
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

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Transforms distribution to standard normal, first. 

        Args:

            x_lower(float): defaults to infinity.
            x_upper(float): defaults to None. 

        Returns:
            p-value drawn from normal distirbution between x_lower and x_upper.
        '''
        mean = self.mean
        std = self.std
        if x_upper is None:
            x_upper = self.randvar

        x_upper = (x_upper - mean) / std
        if x_lower != -np.inf:
            x_lower = (x_lower - mean) / std
        generator = lambda x: 1 / np.sqrt(2 * np.pi) * np.exp(-x**2 / 2)
        return sci.integrate.quad(generator, x_lower, x_upper)[0]

    def confidence_interval(self):
        # find critical values for a given p-value
        pass


class T_distribution(Base):
    '''
    This class contains implementation of the Student's Distribution for calculating the
    probablity density function and cumulative distirbution function. Additionally, 
    a t-table generator is also provided by p-value method. Note that the implementation
    of T(Student's) distribution is defined by beta-functions. 

    Args:
        df(int): degrees of freedom. Defined as d.f. = n-1 where n is the sample size.
        randvar(float): random variable. 
    
    Methods:
        - pdf for probability density function.
        - cdf for cumulative distirbution function.
        - p_value for p value given a random variable x.

    References: 

    - Kruschke JK (2015). Doing Bayesian Data Analysis (2nd ed.). Academic Press. ISBN 9780124058880. OCLC 959632184.
    - Weisstein, Eric W. "Student's t-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Studentst-Distribution.html

    '''
    def __init__(self, df, randvar):
        self.df = df
        self.randvar = randvar

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

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for randvar or plot of the T distribution.
        '''
        df = self.df
        randvar = self.randvar
        generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
            1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distirbution evaluation for some point or plot of the T distribution.
        '''
        df = self.df
        randvar = self.randvar

        def generator(x, df):
            generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
                1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)
            return sci.integrate.quad(generator, -np.inf, x, args=df)[0]

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(randvar, df)

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -∞. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.
            
        Returns:
            p-value of the T distribution evaluated at some random variable.
        '''
        df = self.df
        if x_upper == None:
            x_upper = self.randvar

        generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
            1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)

        return sci.integrate.quad(generator, x_lower, x_upper, args=df)[0]

    def confidence_interval(self):  # for single means and multiple means
        pass


class Cauchy(Base):
    '''
    This class contains methods concerning the Cauchy Distribution.
    
    Args:

        scale(float): pertains to  the scale parameter
        location(float): pertains to the location parameter or median
        x(float): random variable

    Methods:

        - pdf for probability density function
        - cdf for cumulative distribution function
        - p_value for p value

    References:
    - Wikipedia contributors. (2020, November 29). Cauchy distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:01, December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
    - Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    '''
    def __init__(self, x, location, scale):
        self.scale = scale
        self.location = location
        self.x = x

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

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Cauchy distribution.
        '''
        x = self.x
        location = self.location
        scale = self.scale
        generator = lambda x, location, scale: 1 / (np.pi * scale * (1 + (
            (x - location) / scale)**2))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, location, scale) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(x, location, scale)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distirbution evaluation for some point or plot of Cauchy distribution.
        '''
        x = self.x
        location = self.location
        scale = self.scale
        generator = lambda x, location, scale: (1 / np.pi) * np.arctan(
            (x - location) / scale) + 1 / 2
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, location, scale) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(x, location, scale)

    def p_value(self, x_lower=None, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to None. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Cauchy distribution evaluated at some random variable.
        '''
        x = self.x
        location = self.location, scale = self.scale
        generator = lambda x, location, scale: (1 / np.pi) * np.arctan(
            (x - location) / scale) + 1 / 2
        # this messy logic can be improved
        if (x_lower and x_upper) != None:
            return generator(x_lower, location, scale) - generator(
                x_upper, location, scale)
        if x_lower is not None:
            x = x_lower
        if x_upper is not None:
            x = x_upper
        return generator(x, location, scale)

    def confidence_interval(self):
        pass


class F_distribution(Base):
    '''
    This class contains methods concerning the F-distribution. 

    Args:

        - x(float): random variable
        - df1(float): first degrees of freedom
        - df2(float): second degrees of freedom

    Methods:

        - pdf for probability density function
        - cdf for cumulative distribution function
        - p_value for p value
    
    References:
    - Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974). 
    Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.

    - Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
    '''
    def __init__(self, x, df1, df2):
        self.x = x
        self.df1 = df1
        self.df2

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

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of F-distribution.
        '''
        df1 = self.df1
        df2 = self.df2
        randvar = self.x
        generator = lambda x, df1, df2: (1 / ss.beta(
            df1 / 2, df2 / 2)) * np.power(df1 / df2, df1 / 2) * np.power(
                x, df1 / 2 - 1) * np.power(1 +
                                           (df1 / df2) * x, -((df1 + df2) / 2))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df1, df2) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(randvar, df1, df2)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of F-distribution.
        '''
        pass

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the F-distribution evaluated at some random variable.
        '''
        df1 = self.df1
        df2 = self.df2
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x
        pdf_func = lambda x, df1, df2: (1 / ss.beta(
            df1 / 2, df2 / 2)) * np.power(df1 / df2, df1 / 2) * np.power(
                x, df1 / 2 - 1) * np.power(1 +
                                           (df1 / df2) * x, -((df1 + df2) / 2))

        return sci.integrate.quad(pdf_func, x_lower, x_upper,
                                  args=(df1, df2))[0]

    def confidence_interval(self):
        pass


class Chisq_distribution(Base):
    '''
    This class contains methods concerning the Chi-square distribution.

    References:
    - Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    - Wikipedia contributors. (2020, December 13). Chi-square distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:37, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    '''
    def __init__(self, df, x):
        self.x = x
        self.df = df

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
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Chi square-distribution.
    
        '''
        df = self.df
        randvar = self.x
        generator = lambda x, df: (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:

            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Chi square-distribution.
        '''
        df = self.df
        randvar = self.x

        generator = lambda x, df: (1 / ss.gamma(df / 2)) * ss.gammainc(
            df / 2, x / 2) * 2
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(randvar, df)

    def p_val(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi square distribution evaluated at some random variable.
        '''
        x = self.x
        df = self.df
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        pdf_func = lambda x, df: (1 / (np.power(2, df / 2))) * np.power(
            x, df / 2 - 1) * np.exp(-x / 2)

        return sci.integrate.quad(pdf_func, x_lower, x_upper, args=(df))[0] / 2


# check plotting function
class Explonential_distribution(Base):
    '''
    This class contans methods for evaluating Exponential Distirbution. 

    Args:

        - _lambda(float): rate parameter. 
        - x(float): random variable. 

    Methods:

        - pdf for proability density function.
        - cdf for cumulative distribution function.
        - p_value for p value.
    
    References:
    - Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ExponentialDistribution.html
    - Wikipedia contributors. (2020, December 17). Exponential distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:38, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    '''
    def __init__(self, _lambda, x):
        self._lambda = _lambda
        self.x = x

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
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of exponential-distribution.
        '''
        _lambda = self._lambda
        x = self.x

        def generator(_lambda, x):
            if x >= 0:
                return _lambda * np.exp(-(_lambda * x))
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(_lambda, x_i) for x_i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(_lambda, x)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of  exponential distribution.
        '''
        _lambda = self._lambda
        x = self.x

        def generator(x, _lambda):
            if x > 0:
                return 1 - np.exp(-_lambda * x)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(x_i, _lambda) for x_i in x])
            super().plto(x, y, xlim, ylim, xlabel, ylabel)
        return generator(x, _lambda)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Exponential distribution evaluated at some random variable.
        '''
        _lambda = self._lambda
        x = self.x
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = x

        def pdf_func(x, _lambda):
            if x >= 0:
                return _lambda * np.exp(-(_lambda * x))
            return 0

        return sci.integrate.quad(pdf_func, x_lower, x_upper,
                                  args=(_lambda))[0]


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
    '''
    This class contains methods concerning the Pareto Distribution. 

    Args:

        scale(float): scale parameter.
        shape(float): shape parameter.

    Methods

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p values at some random variable x.
    
    References:
    - Barry C. Arnold (1983). Pareto Distributions. International Co-operative Publishing House. ISBN 978-0-89974-012-6.
    - Wikipedia contributors. (2020, December 1). Pareto distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:00, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Pareto_distribution&oldid=991727349
    '''
    def __init__(self, shape, scale, x):
        self.shape = shape
        self.scale = scale
        self.x = x

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
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Pareto-distribution.
        '''
        x_m = self.scale
        alpha = self.shape

        def generator(x, x_m, alpha):
            if x >= x_m:
                return (alpha * x_m**alpha) / np.power(x, alpha + 1)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, x_m, alpha) for i in x])
            super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.x, x_m, alpha)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        '''
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Pareto-distribution.
        '''

        pass

    def p_value(self):
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

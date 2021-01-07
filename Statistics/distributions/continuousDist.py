try:
    import numpy as np
    from math import sqrt
    import scipy as sci
    import scipy.special as ss
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))

# todo = Base: fill between method, implement name mangling 
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
        plt.plot(x, y, "black", alpha=0.5)


class Uniform:
    '''
    This class contains methods concerning the Continuous Uniform Distribution.

    Args: 

        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

    Methods

        - pmf for probability density function
        - cdf for cumulative distribution function
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.

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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(a, b, threshold)  # what does it really say?

    def mean(self):
        '''
        Returns:
            Mean of the Uniform distribution
        '''
        return 1 / 2 * (self.a + self.b)

    def median(self):
        '''
        Returns:
            Median of the Uniform distribution
        '''
        return 1 / 2 * (self.a + self.b)

    def mode(self):
        '''
        Returns:
            Mode of the Uniform distribution. 

        Note that the mode is any value in (a,b)
        '''
        return (self.a, self.b)

    def var(self):
        '''
        Returns:
            Variance of the Uniform distribution
        '''
        return (1 / 12) * (self.b - self.a)**2

    def skewness(self):
        '''
        Returns:
            Skewness of the Uniform distribution
        '''
        return 0

    def kurtosis(self):
        '''
        Returns:
            Kurtosis of the Uniform distribution
        '''
        return -6 / 5


class Normal(Base):
    '''
    This class contains methods concerning the Standard Normal Distribution.

    Args: 

        mean(float): mean of the distribution
        std(float | x>0): standard deviation of the distribution
        randvar(float): random variable

    Methods:

        - pdf for evaluating or plotting probability density function
        - cdf for evaluating or plotting cumulative distribution function
        - p_value returns p-value at randvar.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(self.randvar, mean, std)

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
            either plot of the distirbution or cumulative density evaluation at randvar.
        '''
        generator = lambda mu, sig, x: 1/2*(1+ss.erf((x-mu)/(sig*np.sqrt(2))))
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([generator(self.mean, self.std, x_temp) for x_temp in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.mean, self.std, self.randvar)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Normal distribution evaluated at some random variable.
        '''
        cdf_func = lambda mu, sig, x: 1/2*(1+ss.erf((x-mu)/(sig*np.sqrt(2))))
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return cdf_func(self.mean, self.std, x_upper) - cdf_func(self.mean, self.std, x_lower)
        return cdf_func(self.mean, self.std, self.randvar)

    def confidence_interval(self):
        # find critical values for a given p-value
        pass

    def mean(self):
        '''
        Returns: Mean of the Normal distribution
        '''
        return self.mean

    def median(self):
        '''
        Returns: Median of the Normal distribution
        '''
        return self.mean

    def mode(self):
        '''
        Returns: Mode of the Normal distribution
        '''
        return self.mean

    def var(self):
        '''
        Returns: Variance of the Normal distribution
        '''
        return (self.std)**2

    def skewness(self):
        '''
        Returns: Skewness of the Normal distribution
        '''
        return 0

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Normal distribution
        '''
        return 0

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Normal distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class T_distribution(Base):
    '''
    This class contains implementation of the Student's Distribution for calculating the
    probablity density function and cumulative distirbution function. Additionally, 
    a t-table generator is also provided by p-value method. Note that the implementation
    of T(Student's) distribution is defined by beta-functions. 

    Args:
        df(int): degrees of freedom. 
        randvar(float): random variable. 
    
    Methods:

        - pdf for plotting or evaluating probability density function.
        - cdf for plotting or evaluating cumulative distirbution function.
        - p_value for p value given a random variable x.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References: 

    - Kruschke JK (2015). Doing Bayesian Data Analysis (2nd ed.). Academic Press. ISBN 9780124058880. OCLC 959632184.
    - Weisstein, Eric W. "Student's t-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Studentst-Distribution.html

    '''
    def __init__(self, df, randvar):
        if isinstance(df, int) == False or df<0:
            raise Exception('degrees of freedom(df) should be a whole number. Entered value for df: {}'.format(df))
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None): # cdf definition is not used due to unsupported hypergeometric function 2f1
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(randvar, df)

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.
            
        Returns:
            p-value of the T distribution evaluated at some random variable.
        ''' # normallyt this would be implemented as cdf function from the generalized hypergeometric function
        df = self.df
        if x_upper == None:
            x_upper = self.randvar

        generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
            1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)

        return sci.integrate.quad(generator, x_lower, x_upper, args=df)[0]

    def confidence_interval(self):  # for single means and multiple means
        pass

    def mean(self):
        '''
        Returns:
            Mean of the T-distribution.
        
        	0 for df > 1, otherwise undefined
        '''
        df = self.df
        if df > 1:
            return 0
        return "undefined"

    def median(self):
        '''
        Returns: Median of the T-distribution
        '''
        return 0

    def mode(self):
        '''
        Returns: Mode of the T-distribution
        '''
        return 0

    def var(self):
        '''
        Returns: Variance of the T-distribution

        Note: returns none if it is the case that it is undefined.
        '''
        df = self.df
        if df > 2:
            return df / (df - 2)
        if df > 1 and df <= 2:
            return np.inf
        return None

    def skewness(self):
        '''
        Returns: Skewness of the T-distribution

        Note: returns none if it is the case that it is undefined.
        '''
        df = self.df
        if df > 3:
            return 0
        return None

    def kurtosis(self):
        '''
        Returns: Kurtosis of the T-distribution

        Note: returns none if it is undefined. 
        '''
        df = self.df
        if df > 4:
            return 6 / (df - 4)
        if df > 2 and df <= 4:
            return np.inf
        return None

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the T-distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Cauchy(Base):
    '''
    This class contains methods concerning the Cauchy Distribution.
    
    Args:

        scale(float | x>0): pertains to  the scale parameter
        location(float): pertains to the location parameter or median
        x(float): random variable

    Methods:

        - pdf for plotting or evaluating probability density function
        - cdf for plotting or evaluating cumulative distribution function
        - p_value for p value
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Wikipedia contributors. (2020, November 29). Cauchy distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:01, December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
    - Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    '''
    def __init__(self, x, location, scale):
        if scale<0:
            raise Exception('scale should be greater than 0. Entered value for scale:{}'.format(scale))
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return generator(x, location, scale)

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Cauchy distribution evaluated at some random variable.
        '''
        cdf_func =lambda x, location, scale: (1 / np.pi) * np.arctan((x - location) / scale) + 1 / 2
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return cdf_func(x_upper, self.location, self.scale) - cdf_func(x_lower, self.location, self.scale)
        return cdf_func(self.x, self.location, self.scale)

    def confidence_interval(self):
        pass

    def mean(self):
        '''
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        '''
        return "undefined"

    def median(self):
        '''
        Returns: Median of the Cauchy distribution.
        '''
        return self.location

    def mode(self):
        '''
        Returns: Mode of the Cauchy distribution
        '''
        return self.location

    def var(self):
        '''
        Returns: Variance of the Cauchy distribution. Undefined.
        '''
        return "undefined"

    def skewness(self):
        '''
        Returns: Skewness of the Cauchy distribution. Undefined.
        '''
        return "undefined"

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Cauchy distribution
        '''
        return np.log(4 * np.pi * self.scale)

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Cauchy distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class F_distribution(Base):
    '''
    This class contains methods concerning the F-distribution. 

    Args:

        x(float | [0,infty)): random variable
        df1(int | x>0): first degrees of freedom
        df2(int | x>0): second degrees of freedom

    Methods:

        - pdf for plotting or evaluating probability density function
        - cdf for plotting or evaluating cumulative distribution function
        - p_value for p value
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974). 
    Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.

    - Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
    - NIST SemaTech (n.d.). F-Distribution. Retrived from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm
    '''
    def __init__(self, x, df1, df2):
        if isinstance(df1, int) == False or df1<0:
            raise Exception('degrees of freedom(df) should be a whole number. Entered value for df1: {}'.format(df1))
        if isinstance(df2, int) == False or df2<0:
            raise Exception('degrees of freedom(df) should be a whole number. Entered value for df2: {}'.format(df2))
        if x<0:
            raise Exception('random variable should be greater than 0. Entered value for x:{}'.format(x))

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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
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
        k = self.df2/(self.df2+self.df1*self.x)
        generator = lambda x, df1, df2: 1-ss.betainc(df1/2, df2/2, x)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i,self.df1, self.df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(k,self.df1, self.df2)

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
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        cdf_func = lambda x, df1, df2: 1-ss.betainc(df1/2, df2/2, df2/(df2+df1*x))

        return cdf_func(x_upper, self.df1, self.df2) - cdf_func(x_lower, self.df1, self.df2)

    def confidence_interval(self):
        pass

    def mean(self):
        '''
        Returns: Mean of the F-distribution.
        '''
        if self.df3 > 2:
            return self.df2 / (self.df2 - 2)
        return "undefined"

    def median(self):
        '''
        Returns: Median of the F-distribution. Returns None if the evaluation is currently unsupported.
        '''
        return "unsupported"

    def mode(self):
        '''
        Returns: Mode of the F-distribution. Returns None if undefined.
        '''
        df1 = self.df1
        df2 = self.df2
        if df1 > 2:
            return (df2 * (df1 - 2)) / (df1 * (df2 + 2))
        return "undefined"

    def var(self):
        '''
        Returns: Variance of the F-distribution. Returns None if undefined.
        '''
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return (2 * (df2**2) * (df1 + df2 - 2)) / (df1 * ((df2 - 2)**2) *
                                                       (df2 - 4))
        return "undefined"

    def skewness(self):
        '''
        Returns: Skewness of the F-distribution. Returns None if undefined.
        '''
        df1 = self.df1
        df2 = self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * np.sqrt(8 * (df2 - 4))) / (
                (df2 - 6) * np.sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def kurtosis(self):
        '''
        Returns: Kurtosis of the F-distribution. Returns None if currently unsupported.
        '''
        return "unsupported"

    def print_summary(self):
        '''
        Returns:  summary statistic regarding the F-distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chisq_distribution(Base):
    '''
    This class contains methods concerning the Chi-square distribution.

    Args:

        x(float): random variable.
        df(int): degrees of freedom.

    Methods:
    
        - pdf for evaluating or plotting probability density function.
        - cdf for evaluating or plotting cumulative distribution function.
        - p_value for p value.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    - Wikipedia contributors. (2020, December 13). Chi-square distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:37, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    '''
    def __init__(self, df, x):
        if isinstance(df, int) == False:
            raise Exception('degrees of freedom(df) should be a whole number. Entered value for df: {}'.format(df))
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

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
        generator = lambda x, df:ss.gammainc(df / 2, x / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): p_values of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi square distribution evaluated at some random variable.
        '''
        cdf_func = lambda x, df:ss.gammainc(df / 2, x / 2)
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return cdf_func(x_upper, self.df) - cdf_func(x_lower, self.df)
        return cdf_func(self.randvar, self.df)

    def mean(self):
        '''
        Returns: Mean of the Chi-square distribution.
        '''
        return self.df

    def median(self):
        '''
        Returns: Median of the Chi-square distribution.
        '''
        return self.k * (1 - 2 / (9 * self.k))**3

    def mode(self):
        '''
        Returns: Mode of the Chi-square distribution. Returns None if currently unsupported.
        '''
        return "unsupported"

    def var(self):
        '''
        Returns: Variance of the Chi-square distribution.
        '''
        return 2 * self.df

    def skewness(self):
        '''
        Returns: Skewness of the Chi-square distribution.
        '''
        return np.sqrt(8 / self.df)

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Chi-square distribution.
        '''
        return 12 / self.df

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Chi-square distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chi_distribution(Base):
    '''
    This class contains methods concerning the Chi distribution.

    Args:

        x(float): random variable.
        df(int | x>0): degrees of freedom.

    Methods:
    
        - pdf for evaluating or plotting probability density function.
        - cdf for evaluating or plotting cumulative distribution function.
        - p_value for p value.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Chi Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ChiDistribution.html
    - Wikipedia contributors. (2020, October 16). Chi distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:35, January 2, 2021, from https://en.wikipedia.org/w/index.php?title=Chi_distribution&oldid=983750392
    '''
    def __init__(self, df, x):
        if isinstance(df, int) == False:
            raise Exception('degrees of freedom(df) should be a whole number. Entered value for df: {}'.format(df))
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
            either probability density evaluation for some point or plot of Chi-distribution.
    
        '''
        df = self.df
        randvar = self.x
        generator = lambda x, df: (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

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
            either cumulative distribution evaluation for some point or plot of Chi-distribution.
        '''
        generator = lambda x, df:ss.gammainc(df/2, x**2/2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): p_values of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi distribution evaluated at some random variable.
        '''
        cdf_func = lambda x, df:ss.gammainc(df/2, x**2/2)
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return cdf_func(x_upper, self.df) - cdf_func(x_lower, self.df)
        return cdf_func(self.randvar, self.df)

    def mean(self):
        '''
        Returns: Mean of the Chi distribution.
        '''
        return np.sqrt(2)*ss.gamma((self.df+1)/2)/ss.gamma(self.df/2)

    def median(self):
        '''
        Returns: Median of the Chi distribution.
        '''
        return np.power(self.df*(1-(2/(1*self.df))), 3/2)

    def mode(self):
        '''
        Returns: Mode of the Chi distribution.
        '''
        if self.df>=1:
            return np.sqrt(self.df-1)
        return "undefined"

    def var(self):
        '''
        Returns: Variance of the Chi distribution.
        '''
        return self.df-self.mean()**2

    def skewness(self):
        '''
        Returns: Skewness of the Chi distribution.
        '''
        std = np.sqrt(self.var())
        return (self.mean()-2*self.mean()*std**2)/std**3

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Chi distribution.
        '''
        sk = self.skewness()
        var = self.var()
        mean = self.mean()
        return 2*(1-mean*np.sqrt(var)*sk-var)/var

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Chi distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check plotting function
class Explonential_distribution(Base):
    '''
    This class contans methods for evaluating Exponential Distirbution. 

    Args:

        - _lambda(float | x>0): rate parameter. 
        - x(float | x>0): random variable. 

    Methods:

        - pdf for proability density function.
        - cdf for cumulative distribution function.
        - p_value for p value.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ExponentialDistribution.html
    - Wikipedia contributors. (2020, December 17). Exponential distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:38, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    '''
    def __init__(self, _lambda, x):
        if _lambda<0:
            raise Exception('lambda parameter should be greater than 0. Entered value for _lambda:{}'.format(_lambda))
        if x<0:
            raise Exception('random variable should be greater than 0. Entered value for x:{}'.format(x))
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
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
            raise Exception('x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = x

        def cdf_func(x, _lambda):
            if x > 0:
                return 1 - np.exp(-_lambda * x)
            return 0
        return cdf_func(x_upper, _lambda) - cdf_func(x_lower, _lambda)

    def mean(self):
        '''
        Returns Mean of the Exponential distribution
        '''
        return 1 / self._lambda

    def median(self):
        '''
        Returns Median of the Exponential distribution
        '''
        return np.log(2) / self._lambda

    def mode(self):
        '''
        Returns Mode of the Exponential distribution
        '''
        return 0

    def var(self):
        '''
        Returns Variance of the Exponential distribution
        '''
        return 1 / (self._lambda**2)

    def skewness(self):
        '''
        Returns Skewness of the Exponential distribution
        '''
        return 2

    def kurtosis(self):
        '''
        Returns Kurtosis of the Exponential distribution
        '''
        return 6
    def print_summary(self):
        '''
        Returns: summary statistic regarding the Exponential distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check. add p_value method.
class Gamma_distribution(Base):
    '''
    This class contains methods concerning a variant of Gamma distribution. 

    Args:

        a(float | [0, infty)): shape
        b(float | [0, infty)): scale
        x(float | [0, infty)): random variable

    Methods:

        - pdf for proability density function.
        - cdf for cumulative distribution function.
        - p_value for p value.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Matlab(2020). Gamma Distribution. 
    Retrieved from: https://www.mathworks.com/help/stats/gamma-distribution.html?searchHighlight=gamma%20distribution&s_tid=srchtitle
    '''
    def __init__(self, a, b, x):
        if a<0:
            raise Exception('shape should be greater than 0. Entered value for a:{}'.format(a))
        if b<0:
            raise Exception('scale should be greater than 0. Entered value for b:{}'.format(b))
        if x<0:
            raise Exception('random variable should be greater than 0. Entered value for x:{}'.format(x))
        self.a = a
        self.b = b
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
            either probability density evaluation for some point or plot of Gamma-distribution.
        '''
        generator = lambda a, b, x: (1 / (b**a * ss.gamma(a))) * np.power(
            x, a - 1) * np.exp(-x / b)
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.x)

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
            either cumulative distribution evaluation for some point or plot of Gamma-distribution.
        '''
        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        generator = lambda a, b, x: 1 - ss.gammainc(a, x / b)  
        
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.x)    

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gamma distribution evaluated at some random variable.
        '''
        if x_lower < 0:
            raise Exception('x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = self.x
        cdf_func = lambda a, b, x: 1 - ss.gammainc(a, x / b) 

        return cdf_func(self.a, self.b, x_upper, self._lambda) - cdf_func(self.a, self.b, x_lower, self._lambda)

    def mean(self):
        '''
        Returns: Mean of the Gamma distribution
        '''
        return self.a * self.b

    def median(self):
        '''
        Returns: Median of the Gamma distribution. 
        '''
        return "No simple closed form."

    def mode(self):
        '''
        Returns: Mode of the Gamma distribution
        '''
        return (self.a - 1) * self.b

    def var(self):
        '''
        Returns: Variance of the Gamma distribution
        '''
        return self.a * self.b**2

    def skewness(self):
        '''
        Returns: Skewness of the Gamma distribution
        '''
        return 2 / np.sqrt(self.a)

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Gamma distribution
        '''
        return 6 / self.a

    def print_summary(self):
        '''
        Returns: summary statistic regarding the Gamma distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Pareto(Base):
    '''
    This class contains methods concerning the Pareto Distribution Type 1. 

    Args:

        scale(float | x>0): scale parameter.
        shape(float | x>0): shape parameter.
        x(float | [shape, infty]): random variable.

    Methods

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p values at some random variable x.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Barry C. Arnold (1983). Pareto Distributions. International Co-operative Publishing House. ISBN 978-0-89974-012-6.
    - Wikipedia contributors. (2020, December 1). Pareto distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:00, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Pareto_distribution&oldid=991727349
    '''
    def __init__(self, shape, scale, x):
        if scale<0:
            raise Exception('scale should be greater than 0. Entered value for scale:{}'.format(scale))
        if shape<0:
            raise Exception('shape should be greater than 0. Entered value for shape:{}'.format(shape))
        if x>shape:
            raise Exception('random variable x should be greater than or equal to shape. Entered value for x:{}'.format(x))
        
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
            either probability density evaluation for some point or plot of Pareto distribution.
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
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
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
            either cumulative distribution evaluation for some point or plot of Pareto distribution.
        '''
        x_m = self.scale
        alpha = self.shape

        def generator(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i, x_m, alpha) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.x, x_m, alpha)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Pareto distribution evaluated at some random variable.
        '''
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        def cdf_func(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0
        return cdf_func(x_upper, self.scale, self.alpha)+cdf_func(x_lower, self.scale, self.alpha)

    def mean(self):
        '''
        Returns: Mean of the Pareto distribution.
        '''
        a = self.shape
        x_m = self.scale

        if a <= 1:
            return np.inf
        return (a * x_m) / (a - 1)

    def median(self):
        '''
        Returns: Median of the Pareto distribution.
        '''
        a = self.shape
        x_m = self.scale
        return x_m * np.power(2, 1 / a)

    def mode(self):
        '''
        Returns: Mode of the Pareto distribution.
        '''
        return self.scale

    def var(self):
        '''
        Returns: Variance of the Pareto distribution.
        '''
        a = self.shape
        x_m = self.scale
        if a <= 2:
            return np.inf
        return ((x_m**2) * a) / (((a - 1)**2) * (a - 2))

    def skewness(self):
        '''
        Returns: Skewness of the Pareto distribution. 
        '''
        a = self.shape
        x_m = self.scale
        if a > 3:
            scale = (2 * (1 + a)) / (a - 3)
            return scale * np.sqrt((a - 2) / a)
        return "undefined"

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Pareto distribution. 
        '''
        a = self.shape
        x_m = self.scale
        if a > 4:
            return (6 * (a**3 + a**2 - 6 * a - 2)) / (a * (a - 3) * (a - 4))
        return "undefined"

    def print_summary(self):
        '''
        Returns: summary statistic regarding the Pareto distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# resolve p_value
class Log_normal(Base):
    '''
    This class contains methods concerning the Log Normal Distribution. 

    Args:
        
        randvar(float | [0, infty)): random variable
        mean(float): mean parameter
        std(float | x>0): standard deviation

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Log Normal Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/LogNormalDistribution.html
    - Wikipedia contributors. (2020, December 18). Log-normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:49, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Log-normal_distribution&oldid=994919804
    '''
    def __init__(self, randvar, mean, std):
        if randvar<0:
            raise Exception('random variable should be greater than 0. Entered value for randvar:{}'.format(randvar))
        if std<0:
            raise Exception('random variable should be greater than 0. Entered value for std:{}'.format(std))
        self.randvar = randvar
        self.mean = mean
        self.std = std

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
            either probability density evaluation for some point or plot of Log Normal-distribution.
        '''
        generator = lambda mean, std, x: (1 / (x * std * np.sqrt(
            2 * np.pi))) * np.exp(-(np.log(x - mean)**2) / (2 * std**2))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.mean, self.std, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.mean, self.std, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Log Normal-distribution.
        '''
        generator = lambda mean, std, x:0.5+ 0.5*ss.erfc(-(np.log(x - mean) /
                                                           (std * np.sqrt(2))))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.mean, self.std, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.mean, self.std, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Pareto distribution evaluated at some random variable.
        '''
        cdf_func = lambda mean, std, x:0.5+ 0.5*ss.erfc(-(np.log(x - mean) /
                                                           (std * np.sqrt(2))))
        if x_lower <0:
            raise Exception('x_lower should not be less then 0. X_lower: {}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar

        return cdf_func(self.mean, self.std, x_upper)-cdf_func(self.mean, self.std, x_lower)

    def mean(self):
        '''
        Returns: Mean of the log normal distribution.
        '''
        return np.exp(self.mean + (self.std**2 / 2))

    def median(self):
        '''
        Returns: Median of the log normal distribution.
        '''
        return np.exp(self.mean)

    def mode(self):
        '''
        Returns: Mode of the log normal distribution.
        '''
        return np.exp(self.mean - self.std**2)

    def var(self):
        '''
        Returns: Variance of the log normal distribution.
        '''
        std = self.std
        mean = self.mean
        return (np.exp(std**2) - 1) * np.exp(2 * mean + std**2)

    def skewness(self):
        '''
        Returns: Skewness of the log normal distribution.
        '''
        std = self.std
        mean = self.mean
        return (np.exp(std**2) + 2) * np.sqrt(np.exp(std**2) - 1)

    def kurtosis(self):
        '''
        Returns: Kurtosis of the log normal distribution.
        '''
        std = self.std
        return np.exp(
            4 * std**2) + 2 * np.exp(3 * std**2) + 3 * np.exp(2 * std**2) - 6

    def print_summary(self):
        '''
        Returns: summary statistic regarding the log normal distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# add p_value method, check on ipynb
class Laplace(Base):
    '''
    This class contains methods concerning Laplace Distirbution. 
    Args:
    
        location(float): mean parameter
        scale(float| x>0): standard deviation
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
        - Wikipedia contributors. (2020, December 21). Laplace distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:53, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Laplace_distribution&oldid=995563221
    '''
    def __init__(self, location, scale, randvar):
        if scale<0:
            raise Exception('scale should be greater than 0. Entered value for Scale:{}'.format(scale))
        self.scale = scale
        self.location = location
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
            either probability density evaluation for some point or plot of Laplace distribution.
        '''
        generator = lambda mu, b, x: (1 / (2 * b)) * np.exp(abs(x - mu) / b)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Laplace distribution.
        '''
        generator = lambda mu, b, x: 1 / 2 + ((1 / 2) * np.sign(x - mu) *
                                              (1 - np.exp(abs(x - mu) / b)))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Pareto distribution evaluated at some random variable.
        '''
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        cdf_func = lambda mu, b, x: 1 / 2 + ((1 / 2) * np.sign(x - mu) * (1 - np.exp(abs(x - mu) / b)))
        
        return cdf_func(self.location, self.scale, x_upper)-cdf_func(self.location, self.scale, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Laplace distribution.
        '''
        return self.location

    def median(self):
        '''
        Returns: Median of the Laplace distribution.
        '''
        return self.location

    def mode(self):
        '''
        Returns: Mode of the Laplace distribution.
        '''
        return self.location

    def var(self):
        '''
        Returns: Variance of the Laplace distribution.
        '''
        return 2 * self.scale**2

    def skewness(self):
        '''
        Returns: Skewness of the Laplace distribution.
        '''
        return 0

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Laplace distribution.
        '''
        return 3

    def print_summary(self):
        '''
        Returns: summary statistic regarding the Laplace distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Logistic(Base):
    '''
    This class contains methods concerning Logistic Distirbution. 
    Args:
    
        location(float): mean parameter
        scale(float | x>0): standard deviation
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 
        
    Reference:
    - Wikipedia contributors. (2020, December 12). Logistic distribution. In Wikipedia, The Free Encyclopedia.
     Retrieved 11:14, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Logistic_distribution&oldid=993793195
    '''
    def __init__(self, location, scale, randvar):
        if scale<0:
            raise Exception('scale should be greater than 0. Entered value for Scale:{}'.format(scale))
        self.scale = scale
        self.location = location
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
            either probability density evaluation for some point or plot of Logistic distribution.
        '''
        generator = lambda mu, s, x: np.exp(-(x - mu) / s) / (s * (1 + np.exp(
            -(x - mu) / s))**2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Logistic distribution.
        '''
        generator = lambda mu, s, x: 1 / (1 + np.exp(-(x - mu) / s))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

    def p_value(self, x_lower=-np.inf, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logistic distribution evaluated at some random variable.
        '''
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        cdf_func = lambda mu, s, x: 1 / (1 + np.exp(-(x - mu) / s))
        return cdf_func(self.location, self.scale, x_upper)- cdf_func(self.location, self.scale, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Logistic distribution.
        '''
        return self.location

    def median(self):
        '''
        Returns: Median of the Logistic distribution.
        '''
        return self.location

    def mode(self):
        '''
        Returns: Mode of the Logistic distribution.
        '''
        return self.location

    def var(self):
        '''
        Returns: Variance of the Logistic distribution.
        '''
        return (self.scale**2 * np.pi**2) / 3

    def skewness(self):
        '''
        Returns: Skewness of the Logistic distribution.
        '''
        return 0

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Logistic distribution.
        '''
        return 6 / 5

    def print_summary(self):
        '''
        Retruns: summary statistics of the Logistic distribution.
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Logit_normal(Base):
    '''
    This class contains methods concerning Logit Normal Distirbution. 
    Args:
    
        sq_scale (float): squared scale parameter
        location(float): location parameter
        randvar(float | [0,1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 9). Logit-normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 07:44, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    '''
    def __init__(self, sq_scale, location, randvar):
        if randvar<0 or randvar>1:
            raise Exception('random variable should only be in between (0,1). Entered value: randvar:{}'.format(randvar))
        self.sq_scale = sq_scale
        self.location = location
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
            either probability density evaluation for some point or plot of Logit Normal distribution.
        '''
        generator = lambda mu, sig, x: (1/(sig*np.sqrt(2*np.pi)))*np.exp(-((ss.logit(x)-mu)**2/(2*sig**2)))*(1/(x*(1-x)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.sq_scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.sq_scale, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Logit Normal distribution.
        '''
        generator = lambda mu, sig, x: 1/2*(1+ss.erf((ss.logit(x)-mu)/(np.sqrt(2*sig**2))))
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.sq_scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.sq_scale, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logit distribution evaluated at some random variable.
        '''
        if x_lower<0:
            raise Exception('x_lower should not be less than 0. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        cdf_func = lambda mu, sig, x: 1/2*(1+ss.erf((ss.logit(x)-mu)/(np.sqrt(2*sig**2))))
        return cdf_func(self.location, self.sq_scale, x_upper)-cdf_func(self.location, self.sq_scale, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Logit Normal distribution.
        '''
        return "no analytical solution"

    def median(self):
        '''
        Returns: Median of the Logit Normal distribution.
        '''
        return "unsupported"

    def mode(self):
        '''
        Returns: Mode of the Logit Normal distribution.
        '''
        return "no analytical solution"

    def var(self):
        '''
        Returns: Variance of the Logit Normal distribution.
        '''

        return "no analytical solution"

    def skewness(self):
        '''
        Returns: Skewness of the Logit Normal distribution. 
        '''
        return "unsupported"

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Logit Normal distribution. 
        '''
        return "unsupported"

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Logit Normal distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Weibull(Base):
    '''
    This class contains methods concerning Weibull Distirbution. Also known as Fréchet distribution.
    Args:
    
        shape(float | [0, infty)): mean parameter
        scale(float | [0, infty)): standard deviation
        randvar(float | [0, infty)): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 13). Weibull distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 11:32, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    '''
    def __init__(self, shape, scale, randvar):
        if shape<0 or scale<0 or randvar<0:
            raise Exception('Parameters should not be less than 0. Entered values: shape: {0}, scale{1}, randvar{2}'.format(shape, scale, randvar))
        self.scale = scale
        self.shape = shape
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
            either probability density evaluation for some point or plot of Weibull distribution.
        '''
        def generator(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return (k/_lambda)*(x/_lambda)**(k-1)*np.exp(-(x/_lambda)**k)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.scale, self.shape, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Weibull distribution.
        '''
        def generator(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return 1-np.exp(-(x/_lambda)**k)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.scale, self.shape, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Weilbull distribution evaluated at some random variable.
        '''
        if x_lower<0:
            raise Exception('x_lower should not be less than 0. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        def cdf_func(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return 1-np.exp(-(x/_lambda)**k)
        return cdf_func(self.location, self.shape, x_upper)-cdf_func(self.location, self.shape, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Weibull distribution.
        '''
        return self.scale*ss.gamma(1+(1/self.shape)

    def median(self):
        '''
        Returns: Median of the Weibull distribution.
        '''
        return self.scale*np.power(np.log(2), 1/self.shape)

    def mode(self):
        '''
        Returns: Mode of the Weibull distribution.
        '''
        k = self.shape
        if k>1:
            return self.scale*np.power((k-1)/k, 1/k)
        return 0

    def var(self):
        '''
        Returns: Variance of the Weibull distribution.
        '''
        _lambda = self.scale
        k = self.shape
        return _lambda**2*(((ss.gamma(1+2/k)- ss.gamma(1+1/k)))**2)

    def skewness(self):
        '''
        Returns: Skewness of the Weibull distribution. Returns None i.e. Unsupported.
        '''
        return None

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Weibull distribution. Returns None i.e. Unsupported.
        '''
        return None

    def print_summary(self):
        '''
        Returns: summary statistics of the Weilbull distribution.
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Weilbull_inv(Base):
    '''
    This class contains methods concerning inverse Weilbull or the Fréchet Distirbution. 
    Args:
    
        shape(float | [0, infty)): shape parameter
        scale(float | [0,infty)]): scale parameter
        location(float | (-infty, infty)): location parameter
        randvar(float | randvar > location): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 7). Fréchet distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 07:28, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Fr%C3%A9chet_distribution&oldid=992938143
    '''
    def __init__(self,  shape, scale, location, randvar):
        if shape<0 or scale<0:
            raise Exception('the value of scale and shape should be greater than 0. Entered values scale was:{0}, shape:{1}'.format(scale, shape))
        if randvar<location:
            raise Exception('random variable should be greater than the location parameter. Entered values: randvar: {0}, location:{1}'.format(randvar, location))
        self.shape = shape
        self.scale = scale
        self.location = location
        self.randvar = randvar

    # resolve overflowing issue
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
            either probability density evaluation for some point or plot of Fréchet distribution.
        '''
        generator = lambda a,s,m,x: (a/s)*np.power((x-m)/s, -1-a)*np.exp(-((x-m)/s)**-a)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.shape, self.scale, self.location, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Fréchet distribution.
        '''
        generator =  lambda a,s,m,x: np.exp(-((x-m)/s)**-a)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.shape, self.scale, self.location, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logit distribution evaluated at some random variable.
        '''
        if x_lower<0:
            raise Exception('x_lower should not be less than 0. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        cdf_func = lambda a,s,m,x: np.exp(-((x-m)/s)**-a)
        return cdf_func(self.shape, self.scale, self.location, x_upper)-cdf_func(self.shape, self.scale, self.location, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Fréchet distribution.
        '''
        if self.shape>1:
            return self.location + (self.scale*ss.gamma(1-1/self.shape))
        return np.inf

    def median(self):
        '''
        Returns: Median of the Fréchet distribution.
        '''
        return self.location+(self.scale/(np.power(np.log(2), 1/self.shape)))

    def mode(self):
        '''
        Returns: Mode of the Fréchet distribution.
        '''
        return self.location+self.scale*(self.shape/(1+self.shape))**(1/self.shape)

    def var(self):
        '''
        Returns: Variance of the Fréchet distribution.
        '''
        a = self.shape
        s = self.scale
        if a>2:
            return (s**2)*(ss.gamma(1-2/a)-ss.gamma(1-1/a)**2)
        return np.inf

    def skewness(self):
        '''
        Returns: Skewness of the Fréchet distribution. 
        '''
        a = self.shape
        if a>3:
            return (ss.gamma(1-3/a)-3*ss.gamma(1-2/a)*ss.gamma(1-1/a)+2*ss.gamma(1-1/a)**3)/(np.sqrt((ss.gamma(1-2/a)-ss.gamma(1-1/a)**2)**3))
        return np.inf

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Fréchet distribution. 
        '''
        a = self.shape
        if a>4:
            return -6+((ss.gamma(1-4/a)-4*ss.gamma(1-3/a)*ss.gamma(1-1/a)+3*ss.gamma(1-2/a)**2)/(ss.gamma(1-2/a)-ss.gamma(1-1/a)**2)**2)
        return np.inf

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Fréchet distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Gumbell(Base):
    '''
    This class contains methods concerning Gumbell Distirbution. 
    Args:
    
        location(float): location parameter
        scale(float>0): scale parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 26). Gumbel distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:22, December 29, 2020, from https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    '''
    def __init__(self, location, scale, randvar):
        if scale<0:
            raise Exception('scale parameter should be greater than 0. The value of the scale parameter is: {}'.format(scale))

        self.location = location
        self.scale = scale
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
            either probability density evaluation for some point or plot of Gumbell distribution.
        '''
        def generator(mu, beta, x):
            z = (x-mu)/beta
            return (1/beta)*np.exp(-(z+np.exp(-z)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Gumbell distribution.
        '''
        def generator(mu, beta, x):
            return np.exp(-np.exp(-(x-mu)/beta))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

    def p_value(self):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gumbell distribution evaluated at some random variable.
        '''
        return "currently unsupported"

    def mean(self):
        '''
        Returns: Mean of the Gumbell distribution.
        '''
        return self.location+(self.scale*np.euler_gamma)

    def median(self):
        '''
        Returns: Median of the Gumbell distribution.
        '''
        return self.location - (self.scale*np.log(np.log(2)))

    def mode(self):
        '''
        Returns: Mode of the Gumbell distribution.
        '''
        return self.location

    def var(self):
        '''
        Returns: Variance of the Gumbell distribution.
        '''
        return (np.pi**2/6)*self.scale**2

    def skewness(self):
        '''
        Returns: Skewness of the Gumbell distribution. 
        '''
        return 1.14

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Gumbell distribution. 
        '''
        return 12/5

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Gumbell distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Arcsine(Base):
    '''
    This class contains methods concerning Arcsine Distirbution. 
    Args:
    
        randvar(float in [0, 1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, October 30). Arcsine distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:19, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    '''
    def __init__(self, randvar):
        if randvar>0 or randvar>1:
            raise Exception('random variable should have values between [0,1]. The value of randvar was: {}'.format(randvar))
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
            either probability density evaluation for some point or plot of Arcsine distribution.
        '''
        generator = lambda x: 1/(np.pi*np.sqrt(x*(1-x)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Arcsine distribution.
        '''
        generator = lambda x: (2/np.pi)*np.arcsin(np.sqrt(x))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.location, self.scale, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Arcsine distribution evaluated at some random variable.
        '''
        if x_lower<0 or x>1:
            raise Exception('x_lower should not be less than 0 or greater than 1. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        cdf_func = lambda x: (2/np.pi)*np.arcsin(np.sqrt(x))
        return cdf_func(self.location, self.scale, x_upper)-cdf_func(self.location, self.scale, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Arcsine distribution.
        '''
        return 1/2

    def median(self):
        '''
        Returns: Median of the Arcsine distribution.
        '''
        return 1/2

    def mode(self):
        '''
        Returns: Mode of the Arcsine distribution. Mode is within the set {0,1}
        '''
        return {0,1}
    def var(self):
        '''
        Returns: Variance of the Arcsine distribution.
        '''
        return 1/8
    def skewness(self):
        '''
        Returns: Skewness of the Arcsine distribution. 
        '''
        return 0

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Arcsine distribution. 
        '''
        return 3/2

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Arcsine distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Triangular(Base):
    '''
    This class contains methods concerning Triangular Distirbution. 
    Args:
    
        a(float): lower limit
        b(float | a<b): upper limit
        c(float| a≤c≤b): mode
        randvar(float | a≤randvar≤b): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 19). Triangular distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:41, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Triangular_distribution&oldid=995101682
    '''
    def __init__(self, a,b,c, randvar):
        if a>b:
            raise Exception('lower limit(a) should be less than upper limit(b).')
        if a>c and c>b:
            raise Exception('lower limit(a) should be less than or equal to mode(c) where c is less than or equal to upper limit(b).')
        self.a = a
        self.b = b
        self.c = c
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
            either probability density evaluation for some point or plot of Triangular distribution.
        '''
        def generator(a,b,c,x):
            if x<a:
                return 0
            if a<=x and x<c:
                return (2*(x-a))/((b-a)*(c-a))
            if x == c:
                return 2/(b-a)
            if c<x and x<=b:
                return (2*(b-x))/((b-a)((b-c)))
            if b<x:
                return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.c, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Triangular distribution.
        '''
        def generator(a,b,c,x):
            if x<=a:
                return 0
            if a<x and x<=c:
                return ((x-a)**2)/((b-a)*(c-a))
            if c<x and x<b:
                return 1 - ((b-x)**2)/((b-c)*(b-c))
            if b<=x:
                return 1
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.c, self.randvar)

    def p_value(self, x_lower=0, x_upper=None):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Triangular distribution evaluated at some random variable.
        '''
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        def cdf_func(a,b,c,x):
            if x<=a:
                return 0
            if a<x and x<=c:
                return ((x-a)**2)/((b-a)*(c-a))
            if c<x and x<b:
                return 1 - ((b-x)**2)/((b-c)*(b-c))
            if b<=x:
                return 1
        return cdf_func(self.a, self.b, self.c, x_upper)-cdf_func(self.a, self.b, self.c, x_lower)

    def mean(self):
        '''
        Returns: Mean of the Triangular distribution.
        '''
        return (self.a+self.b+self.c)/3

    def median(self):
        '''
        Returns: Median of the Triangular distribution.
        '''
        a = self.a
        b = self.b
        c = self.c
        if c >= (a+b)/2:
            return a + np.sqrt(((b-a)*(c-a))/2)
        if c <= (a+b)/2:
            return b + np.sqrt((b-a)*(b-c)/2)

    def mode(self):
        '''
        Returns: Mode of the Triangular distribution.
        '''
        return self.c

    def var(self):
        '''
        Returns: Variance of the Triangular distribution.
        '''
        a = self.a
        b = self.b
        c = self.c
        return (1/18)*(a**2+b**2+c**2-a*b-a*c-b*c)

    def skewness(self):
        '''
        Returns: Skewness of the Triangular distribution. 
        '''
        a = self.a
        b = self.b
        c = self.c
        return (np.sqrt(2)*(a+b-2*c)((2*a-b-c)*(a-2*b+c)))/(5*(a**2+b**2+c**2-a*b-a*c-b*c)**(3/2))

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Triangular distribution. 
        '''
        return -3/5

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Triangular distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Trapezoidal(Base):
    '''
    This class contains methods concerning Trapezoidal Distirbution. 
    Args:
    
        a(float | a<d): lower bound
        b(float | a≤b<c): level start
        c(float | b<c≤d): level end
        d(float | c≤d): upper bound
        randvar(float | a≤randvar≤d): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - p_value for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - print_summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
    '''
    def __init__(self, a,b,c,d, randvar):
        if a>d:
            raise Exception('lower bound(a) should be less than upper bound(d).')
        if a>b or b>=c:
            raise Exception('lower bound(a) should be less then or equal to level start (b) where (b) is less than level end(c).')
        if b>=c or c>d:
            raise Exception('level start(b) should be less then level end(c) where (c) is less then or equal to upper bound (d).')
        if c>d:
            raise Exception('level end(c) should be less than or equal to upper bound(d)')

        self.a = a
        self.b = b
        self.c = c
        self.d = d
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
            either probability density evaluation for some point or plot of Trapezoidal distribution.
        '''
        def generator(a,b,c,d,x):
            if a<=x and x<b:
                return (2/(d+c-a-b))*(x-a)/(b-a)
            if b<=x and x<c:
                return (2/(d+c-a-b))
            if c<=x and x<=d:
                return (2/(d+c-a-b))*(d-x)/(d-c)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.a, self.b, self.c, self.d, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.c, self.d, self.randvar)

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
            either cumulative distribution evaluation for some point or plot of Trapezoidal distribution.
        '''
        def generator(a,b,c,d,x):
            if a<=x and x<b:
                return (x-a)**2/((b-a)*(d+c-a-b))
            if b<=x and x<c:
                return (2*x-a-b)/(d+c-a-b)
            if c<=x and x<=d:
                return 1- (d-x)**2/((d+c-a-b)*(d-c))
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([generator(self.a, self.b, self.c, self.d, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return generator(self.a, self.b, self.c, self.d, self.randvar)

    def p_value(self):
        '''
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Trapezoidal distribution evaluated at some random variable.
        '''
        return "currently unsupported"

    def mean(self):
        '''
        Returns: Mean of the Trapezoidal distribution.
        '''
        return (self.a+self.b+self.c)/3

    def median(self):
        '''
        Returns: Median of the Trapezoidal distribution. Currently Unsupported.
        '''
        return "currently unssuported."

    def mode(self):
        '''
        Returns: Mode of the Trapezoidal distribution. Currently Unsupported.
        '''
        return "currently unssuported."

    def var(self):
        '''
        Returns: Variance of the Trapezoidal distribution. Currently Unsupported. 
        '''
        return "currently unssuported."

    def skewness(self):
        '''
        Returns: Skewness of the Trapezoidal distribution. Currently Unsupported.
        '''
        return "currently unssuported."

    def kurtosis(self):
        '''
        Returns: Kurtosis of the Trapezoidal distribution. Currently Unsupported.
        '''
        return "currently unssuported."

    def print_summary(self):
        '''
        Returns: Summary statistic regarding the Trapezoidal distribution
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class ARGUS(Base):
#     '''
#     This class contains methods concerning ARGUS Distirbution. 
#     Args:
    
#         a(float|a<b): supported parameters
#         b(float): supported parameters
#         randvar(float | [a,b]): random variable

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - p_value for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - print_summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia. 
#     Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
#     '''
#     def __init__(self, a,b, randvar):
#         if a>b:
#             raise Exception('lower bound(a) should be less than upper bound(b).')
#         self.a = a
#         self.b = b
#         self.randvar = randvar

#     def pdf(self,
#             plot=False,
#             interval=1,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either probability density evaluation for some point or plot of ARGUS distribution.
#         '''
#         def generator(a,b,c,d,x):
#             if a<=x and x<b:
#                 return (2/(d+c-a-b))*(x-a)/(b-a)
#             if b<=x and x<c:
#                 return (2/(d+c-a-b))
#             if c<=x and x<=d:
#                 return (2/(d+c-a-b))*(d-x)/(d-c)

#         if plot == True:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([generator(self.a, self.b, self.c, self.d, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return generator(self.a, self.b, self.c, self.d, self.randvar)

#     def cdf(self,
#             plot=False,
#             interval=1,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         '''
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either cumulative distribution evaluation for some point or plot of ARGUS distribution.
#         '''
#         def generator(a,b,c,d,x):
#             if a<=x and x<b:
#                 return (x-a)**2/((b-a)*(d+c-a-b))
#             if b<=x and x<c:
#                 return (2*x-a-b)/(d+c-a-b)
#             if c<=x and x<=d:
#                 return 1- (d-x)**2/((d+c-a-b)*(d-c))
                
#         if plot == True:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([generator(self.a, self.b, self.c, self.d, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return generator(self.a, self.b, self.c, self.d, self.randvar)

#     def p_value(self):
#         '''
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the Pareto distribution evaluated at some random variable.
#         '''
#         return "currently unsupported"

#     def mean(self):
#         '''
#         Returns: Mean of the ARGUS distribution.
#         '''
#         return (self.a+self.b+self.c)/3

#     def median(self):
#         '''
#         Returns: Median of the ARGUS distribution. Currently Unsupported.
#         '''
#         return "currently unssuported."

#     def mode(self):
#         '''
#         Returns: Mode of the ARGUS distribution. Currently Unsupported.
#         '''
#         return "currently unssuported."

#     def var(self):
#         '''
#         Returns: Variance of the ARGUS distribution. Currently Unsupported. 
#         '''
#         return "currently unssuported."

#     def skewness(self):
#         '''
#         Returns: Skewness of the ARGUS distribution. 
#         '''
#         # return 

#     def kurtosis(self):
#         '''
#         Returns: Kurtosis of the ARGUS distribution. Currently Unsupported.
#         '''
#         return "currently unssuported."

#     def print_summary(self):
#         '''
#         Returns: Summary statistic regarding the ARGUS distribution
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
try:
    import numpy as np
    from math import sqrt, pow, log
    import scipy as sci
    import scipy.special as ss
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))

# todo = Base: remove unsupported method from class implementation, implement name mangling 

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

    def logpdf(self, pdf):
        return np.log(pdf)

    def logcdf(self, cdf):
        return np.log(cdf)

    def pvalue(self):
        return "unsupported"

    def confidence_interval(self):
        return "currently unsupported"

    def rvs(self): # (adaptive) rejection sampling implementation
        """
        returns random variate samples default (unsupported)
        """
        return "currently unsupported"

    def mean(self):
        """
        returns mean default (unsupported)
        """
        return "unsupported"

    def median(self):
        """
        returns median default (unsupported)
        """
        return "unsupported"

    def mode(self):
        """
        returns mode default (unsupported)
        """
        return "unsupported"

    def var(self):
        """
        returns variance default (unsupported)
        """
        return "unsupported"

    def std(self):
        """
        returns the std default (undefined)
        """
        return "undefined"
    
    def skewness(self):
        """
        returns skewness default (unsupported)
        """
        return "unsupported"

    def kurtosis(self):
        """
        returns kurtosis default (unsupported)
        """
        return "unsupported"
    
    def entropy(self):
        """
        returns entropy default (unsupported)
        """
        return "unsupported"
    
    # special functions for ϕ(x), and Φ(x) functions: should this be reorganized?
    def std_normal_pdf(self, x):
        return np.exp(-pow(x,2)/2)/sqrt(2*np.pi)
    
    def std_normal_cdf(self, x):
        return sci.integrate.quad(self.std_normal_pdf, -np.inf, x)

# class bounded(Base):
#     pass

# class semi_infinite(Base):
#     pass

# class real_line(Base):
#     pass

# class varying_support(Base):
#     pass

class Uniform:
    """
    This class contains methods concerning the Continuous Uniform Distribution.

    Args: 

        a(int): lower limit of the distribution
        b(int): upper limit of the distribution

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Referene:
    - Weisstein, Eric W. "Uniform Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/UniformDistribution.html
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def pdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        """
        Args:

            plot (bool): returns plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        _generator = lambda a, b, x: 1 / (b - a) if a <= x and x <= b else 0
        if plot == True:
            x = np.linspace(a, b, threshold)
            y = np.array([_generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(a, b, np.abs(b - a))

    def cdf(self, plot=False, xlim=None, ylim=None, xlabel=None, ylabel=None):
        """
        Args:

            plot (bool): returns plot if true. 
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        Returns:
            either plot of the distribution or probability density evaluation at a to b.
        """
        a = self.a
        b = self.b
        threshold = b - a

        def _generator(a, b, x):
            if x < a:
                return 0
            if (a <= x and x <= b):
                return (x - a) / (b - a)
            if x > b:
                return 1

        if plot == True:
            x = np.linspace(a, b, threshold)
            y = np.array([_generator(a, b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(a, b, threshold)  # what does it really say?

    def mean(self):
        """
        Returns: Mean of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def median(self):
        """
        Returns: Median of the Uniform distribution.
        """
        return 1 / 2 * (self.a + self.b)

    def mode(self):
        """
        Returns: Mode of the Uniform distribution. 

        Note that the mode is any value in (a,b)
        """
        return (self.a, self.b)

    def var(self):
        """
        Returns: Variance of the Uniform distribution.
        """
        return (1 / 12) * (self.b - self.a)**2
    def std(self):
        """
        Returns: Standard deviation of the Uniform distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Uniform distribution.
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Uniform distribution.
        """
        return -6 / 5

    def entropy(self):
        """
        Returns: entropy of uniform Distirbution.
        
        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return np.log(self.b-self-a)

    def summary(self):
        """
        Returns: Summary statistic regarding the Uniform distribution.
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Normal(Base):
    """
    This class contains methods concerning the Standard Normal Distribution.

    Args: 

        mean(float): mean of the distribution
        std(float | x>0): standard deviation of the distribution
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Wikipedia contributors. (2020, December 19). Normal distribution. In Wikipedia, The Free Encyclopedia. Retrieved 10:44, 
    December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Normal_distribution&oldid=995237372
    - Weisstein, Eric W. "Normal Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/NormalDistribution.html

    """
    def __init__(self, x, mean=0, std_val=1):
        self.mean_val = mean
        self.std_val = std_val
        self.randvar = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        mean = self.mean_val
        std = self.std_val
        _generator = lambda mean, std, x: np.power(
            1 / (std * np.sqrt(2 * np.pi)), np.exp(((x - mean) / 2 * std)**2))
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(mean, std, x_temp) for x_temp in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(mean, std, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):  
        """
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
        """
        _generator = lambda mu, sig, x: 1/2*(1+ss.erf((x-mu)/(sig*np.sqrt(2))))
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(self.mean_val, self.std_val, x_temp) for x_temp in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.mean_val, self.std_val, self.randvar)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Normal distribution evaluated at some random variable.
        """
        _cdf_def = lambda mu, sig, x: 1/2*(1+ss.erf((x-mu)/(sig*np.sqrt(2))))
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(self.mean_val, self.std_val, x_upper) - _cdf_def(self.mean, self.std_val, x_lower)
        return _cdf_def(self.mean_val, self.std_val, self.randvar)

    def confidence_interval(self):
        # find critical values for a given p-value
        pass

    def mean(self):
        """
        Returns: Mean of the Normal distribution
        """
        return self.mean_val

    def median(self):
        """
        Returns: Median of the Normal distribution
        """
        return self.mean_val

    def mode(self):
        """
        Returns: Mode of the Normal distribution
        """
        return self.mean_val

    def var(self):
        """
        Returns: Variance of the Normal distribution
        """
        return pow(self.std_val,2)

    def std(self):
        """
        Returns: Standard deviation of the Normal distribution
        """
        return self.std_val

    def skewness(self):
        """
        Returns: Skewness of the Normal distribution
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Normal distribution
        """
        return 0

    def entropy(self):
        """
        Returns: differential entropy of the Normal distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return np.log(self.std*sqrt(2*np.pi*np.e))

    def summary(self):
        """
        Returns: Summary statistic regarding the Normal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class T(Base):
    """
    This class contains implementation of the Student's Distribution for calculating the
    probablity density function and cumulative distirbution function. Additionally, 
    a t-table _generator is also provided by p-value method. Note that the implementation
    of T(Student's) distribution is defined by beta-functions. 

    Args:
        df(int): degrees of freedom. 
        randvar(float): random variable. 
    
    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References: 

    - Kruschke JK (2015). Doing Bayesian Data Analysis (2nd ed.). Academic Press. ISBN 9780124058880. OCLC 959632184.
    - Weisstein, Eric W. "Student's t-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Studentst-Distribution.html

    """
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
        """
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
        """
        df = self.df
        randvar = self.randvar
        _generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
            1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None): # cdf definition is not used due to unsupported hypergeometric function 2f1
        """
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
        """
        df = self.df
        randvar = self.randvar

        def _generator(x, df):
            _generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
                1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)
            return sci.integrate.quad(_generator, -np.inf, x, args=df)[0]

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.
            
        Returns:
            p-value of the T distribution evaluated at some random variable.
        """ # normallyt this would be implemented as cdf function from the generalized hypergeometric function
        df = self.df
        if x_upper == None:
            x_upper = self.randvar

        _generator = lambda x, df: (1 / (np.sqrt(df) * ss.beta(
            1 / 2, df / 2))) * np.power((1 + (x**2 / df)), -(df + 1) / 2)

        return sci.integrate.quad(_generator, x_lower, x_upper, args=df)[0]

    def confidence_interval(self):  # for single means and multiple means
        pass

    def mean(self):
        """
        Returns:
            Mean of the T-distribution.
        
        	0 for df > 1, otherwise undefined
        """
        df = self.df
        if df > 1:
            return 0
        return "undefined"

    def median(self):
        """
        Returns: Median of the T-distribution
        """
        return 0

    def mode(self):
        """
        Returns: Mode of the T-distribution
        """
        return 0

    def var(self):
        """
        Returns: Variance of the T-distribution
        """
        df = self.df
        if df > 2:
            return df / (df - 2)
        if df > 1 and df <= 2:
            return np.inf
        return "undefined"
    
    def std(self):
        """
        Returns: Standard Deviation of the T-distribution
        """
        if self.var()=="undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the T-distribution
        """
        df = self.df
        if df > 3:
            return 0
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the T-distribution
        """
        df = self.df
        if df > 4:
            return 6 / (df - 4)
        if df > 2 and df <= 4:
            return np.inf
        return "undefined"

    def entropy(self):
        """
        Returns: differential entropy of T-distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return ((df+1)/2)*(ss.digamma((df+1)/2)-ss.digamma(df/2))+np.log(sqrt(df)*ss.beta(df/2, 1/2))

    def summary(self):
        """
        Returns: Summary statistic regarding the T-distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Cauchy(Base):
    """
    This class contains methods concerning the Cauchy Distribution.
    
    Args:

        scale(float | x>0): pertains to  the scale parameter
        location(float): pertains to the location parameter or median
        x(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Wikipedia contributors. (2020, November 29). Cauchy distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:01, December 22, 2020, from https://en.wikipedia.org/w/index.php?title=Cauchy_distribution&oldid=991234690
    - Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/CauchyDistribution.html
    """
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
        """
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
        """
        x = self.x
        location = self.location
        scale = self.scale
        _generator = lambda x, location, scale: 1 / (np.pi * scale * (1 + (
            (x - location) / scale)**2))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, location, scale) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(x, location, scale)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        x = self.x
        location = self.location
        scale = self.scale
        _generator = lambda x, location, scale: (1 / np.pi) * np.arctan(
            (x - location) / scale) + 1 / 2
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, location, scale) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(x, location, scale)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. Defines the upper value of the distribution. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Cauchy distribution evaluated at some random variable.
        """
        _cdf_def =lambda x, location, scale: (1 / np.pi) * np.arctan((x - location) / scale) + 1 / 2
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(x_upper, self.location, self.scale) - _cdf_def(x_lower, self.location, self.scale)
        return _cdf_def(self.x, self.location, self.scale)

    def confidence_interval(self):
        pass

    def mean(self):
        """
        Returns: Mean of the Cauchy distribution. Mean is Undefined.
        """
        return "undefined"

    def median(self):
        """
        Returns: Median of the Cauchy distribution.
        """
        return self.location

    def mode(self):
        """
        Returns: Mode of the Cauchy distribution
        """
        return self.location

    def var(self):
        """
        Returns: Variance of the Cauchy distribution. 
        """
        return "undefined"
    
    def std(self):
        """
        Returns: Standard Deviation of the Cauchy Distribution.
        """
        return "undefined"

    def skewness(self):
        """
        Returns: Skewness of the Cauchy distribution. 
        """
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Cauchy distribution
        """
        return np.log(4 * np.pi * self.scale)

    def entropy(self):
        """
        Returns: differential entropy of the Cauchy distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return np.log10(4*np.pi*self.scale)

    def summary(self):
        """
        Returns: Summary statistic regarding the Cauchy distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class F(Base):
    """
    This class contains methods concerning the F-distribution. 

    Args:

        x(float | [0,infty)): random variable
        df1(int | x>0): first degrees of freedom
        df2(int | x>0): second degrees of freedom

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Mood, Alexander; Franklin A. Graybill; Duane C. Boes (1974). 
    Introduction to the Theory of Statistics (Third ed.). McGraw-Hill. pp. 246–249. ISBN 0-07-042864-6.

    - Weisstein, Eric W. "F-Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/F-Distribution.html
    - NIST SemaTech (n.d.). F-Distribution. Retrived from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm
    """
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
        """
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
        """
        df1 = self.df1
        df2 = self.df2
        randvar = self.x
        _generator = lambda x, df1, df2: (1 / ss.beta(
            df1 / 2, df2 / 2)) * np.power(df1 / df2, df1 / 2) * np.power(
                x, df1 / 2 - 1) * np.power(1 +
                                           (df1 / df2) * x, -((df1 + df2) / 2))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df1, df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(randvar, df1, df2)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        k = self.df2/(self.df2+self.df1*self.x)
        _generator = lambda x, df1, df2: 1-ss.betainc(df1/2, df2/2, x)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i,self.df1, self.df2) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(k,self.df1, self.df2)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the F-distribution evaluated at some random variable.
        """
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        _cdf_def = lambda x, df1, df2: 1-ss.betainc(df1/2, df2/2, df2/(df2+df1*x))

        return _cdf_def(x_upper, self.df1, self.df2) - _cdf_def(x_lower, self.df1, self.df2)

    def confidence_interval(self):
        pass

    def mean(self):
        """
        Returns: Mean of the F-distribution.
        """
        if self.df3 > 2:
            return self.df2 / (self.df2 - 2)
        return "undefined"

    def mode(self):
        """
        Returns: Mode of the F-distribution. 
        """
        df1 = self.df1
        df2 = self.df2
        if df1 > 2:
            return (df2 * (df1 - 2)) / (df1 * (df2 + 2))
        return "undefined"

    def var(self):
        """
        Returns: Variance of the F-distribution.
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 4:
            return (2 * (df2**2) * (df1 + df2 - 2)) / (df1 * ((df2 - 2)**2) *
                                                       (df2 - 4))
        return "undefined"
    
    def std(self):
        """
        Returns: Standard deviation of the F-distribution.
        """
        if self.var()=="undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the F-distribution. 
        """
        df1 = self.df1
        df2 = self.df2
        if df2 > 6:
            return ((2 * df1 + df2 - 2) * np.sqrt(8 * (df2 - 4))) / (
                (df2 - 6) * np.sqrt(df1 * (df1 + df2 - 2)))
        return "undefined"

    def entropy(self):
        """
        Returns: differential entropy of F-distribution. 

        Reference: Lazo, A.V.; Rathie, P. (1978). "On the entropy of continuous probability distributions". IEEE Transactions on Information Theory
        """
        df1 = self.df1; df2 = self.df2
        return np.log(ss.gamma(df1/2))+np.log(ss.gamma(df2/2))-np.log(ss.gamma((df1+df2)/2))+(1-df1/2)*ss.digamma(1+df1/2)-(1-df2/2)*ss.digamma(1+df2/2)+(df1+df2)/2*ss.digamma((df1+df2)/2)+np.log(df1/df2)

    def summary(self):
        """
        Returns:  summary statistic regarding the F-distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chisq(Base):
    """
    This class contains methods concerning the Chi-square distribution.

    Args:

        x(float): random variable.
        df(int): degrees of freedom.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/Chi-SquaredDistribution.html
    - Wikipedia contributors. (2020, December 13). Chi-square distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:37, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Chi-square_distribution&oldid=994056539
    """
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
        """
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
    
        """
        df = self.df
        randvar = self.x
        _generator = lambda x, df: (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda x, df:ss.gammainc(df / 2, x / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi square distribution evaluated at some random variable.
        """
        _cdf_def = lambda x, df:ss.gammainc(df / 2, x / 2)
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(x_upper, self.df) - _cdf_def(x_lower, self.df)
        return _cdf_def(self.randvar, self.df)

    def mean(self):
        """
        Returns: Mean of the Chi-square distribution.
        """
        return self.df

    def median(self):
        """
        Returns: Median of the Chi-square distribution.
        """
        return self.k * (1 - 2 / (9 * self.k))**3

    def var(self):
        """
        Returns: Variance of the Chi-square distribution.
        """
        return 2 * self.df
    
    def std(self):
        """
        Returns: Standard deviation of the Chi-square distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Chi-square distribution.
        """
        return np.sqrt(8 / self.df)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Chi-square distribution.
        """
        return 12 / self.df

    def entropy(self):
        """
        Returns: differential entropy of Chi-square distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return df/2+np.log(2*ss.gamma(df/2))+(1-df/2)*ss.digamma(df/2)

    def summary(self):
        """
        Returns: Summary statistic regarding the Chi-square distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chi(Base):
    """
    This class contains methods concerning the Chi distribution.

    Args:

        x(float): random variable.
        df(int | x>0): degrees of freedom.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Chi Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ChiDistribution.html
    - Wikipedia contributors. (2020, October 16). Chi distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:35, January 2, 2021, from https://en.wikipedia.org/w/index.php?title=Chi_distribution&oldid=983750392
    """
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
        """
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
    
        """
        df = self.df
        randvar = self.x
        _generator = lambda x, df: (1 / (np.power(2, (df / 2) - 1) * ss.gamma(
            df / 2))) * np.power(x, df - 1) * np.exp(-x**2 / 2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(randvar, df)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda x, df:ss.gammainc(df/2, x**2/2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, self.df) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar, self.df)

    def p_val(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to -np.inf. Defines the lower value of the distribution. Optional.
            x_upper(float | x_upper>x_lower): defaults to None. If not defined defaults to random variable x. Optional.
            args(list of float): pvalues of each elements from the list

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Chi distribution evaluated at some random variable.
        """
        _cdf_def = lambda x, df:ss.gammainc(df/2, x**2/2)
        if x_upper != None:
            if x_lower>x_upper:
                raise Exception('x_lower should be less than x_upper.')
            return _cdf_def(x_upper, self.df) - _cdf_def(x_lower, self.df)
        return _cdf_def(self.randvar, self.df)

    def mean(self):
        """
        Returns: Mean of the Chi distribution.
        """
        return np.sqrt(2)*ss.gamma((self.df+1)/2)/ss.gamma(self.df/2)

    def median(self):
        """
        Returns: Median of the Chi distribution.
        """
        return np.power(self.df*(1-(2/(1*self.df))), 3/2)

    def mode(self):
        """
        Returns: Mode of the Chi distribution.
        """
        if self.df>=1:
            return np.sqrt(self.df-1)
        return "undefined"

    def var(self):
        """
        Returns: Variance of the Chi distribution.
        """
        return pow(self.df-self.mean(),2)
    
    def std(self):
        """
        Returns: Standard deviation of the Chi distribution.
        """
        return self.df-self.mean()

    def skewness(self):
        """
        Returns: Skewness of the Chi distribution.
        """
        std = np.sqrt(self.var())
        return (self.mean()-2*self.mean()*std**2)/std**3

    def kurtosis(self):
        """
        Returns: Kurtosis of the Chi distribution.
        """
        sk = self.skewness()
        var = self.var()
        mean = self.mean()
        return 2*(1-mean*np.sqrt(var)*sk-var)/var

    def entropy(self):
        """
        Returns: differential entropy of Chi distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        df = self.df
        return np.log(ss.gamma(df/2)/sqrt(2))-(df-1)/2*ss.digamma(df/2)+df/2 

    def summary(self):
        """
        Returns: Summary statistic regarding the Chi distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check plotting function
class Explonential(Base):
    """
    This class contans methods for evaluating Exponential Distirbution. 

    Args:

        - lambda_(float | x>0): rate parameter. 
        - x(float | x>0): random variable. 

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Exponential Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ExponentialDistribution.html
    - Wikipedia contributors. (2020, December 17). Exponential distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:38, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Exponential_distribution&oldid=994779060
    """
    def __init__(self, lambda_, x):
        if lambda_<0:
            raise Exception('lambda parameter should be greater than 0. Entered value for lambda_:{}'.format(lambda_))
        if x<0:
            raise Exception('random variable should be greater than 0. Entered value for x:{}'.format(x))
        self.lambda_ = lambda_
        self.x = x

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        lambda_ = self.lambda_
        x = self.x

        def _generator(lambda_, x):
            if x >= 0:
                return lambda_ * np.exp(-(lambda_ * x))
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(lambda_, x_i) for x_i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(lambda_, x)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        lambda_ = self.lambda_
        x = self.x

        def _generator(x, lambda_):
            if x > 0:
                return 1 - np.exp(-lambda_ * x)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(x_i, lambda_) for x_i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(x, lambda_)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Exponential distribution evaluated at some random variable.
        """
        lambda_ = self.lambda_
        x = self.x
        if x_lower < 0:
            raise Exception('x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = x

        def _cdf_def(x, lambda_):
            if x > 0:
                return 1 - np.exp(-lambda_ * x)
            return 0
        return _cdf_def(x_upper, lambda_) - _cdf_def(x_lower, lambda_)

    def mean(self):
        """
        Returns: Mean of the Exponential distribution
        """
        return 1 / self.lambda_

    def median(self):
        """
        Returns: Median of the Exponential distribution
        """
        return np.log(2) / self.lambda_

    def mode(self):
        """
        Returns: Mode of the Exponential distribution
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Exponential distribution
        """
        return 1 / pow(self.lambda_,2)
    
    def std(self):
        """
        Returns: Standard deviation of the Exponential distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Exponential distribution
        """
        return 2

    def kurtosis(self):
        """
        Returns: Kurtosis of the Exponential distribution
        """
        return 6

    def entorpy(self):
        """
        Returns: differential entropy of the Exponential distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1-np.log(self.lambda_)

    def summary(self):
        """
        Returns: summary statistic regarding the Exponential distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# check. add pvalue method.
class Gamma(Base):
    """
    This class contains methods concerning a variant of Gamma distribution. 

    Args:

        a(float | [0, infty)): shape
        b(float | [0, infty)): scale
        x(float | [0, infty)): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Matlab(2020). Gamma Distribution. 
    Retrieved from: https://www.mathworks.com/help/stats/gamma-distribution.html?searchHighlight=gamma%20distribution&s_tid=srchtitle
    """
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
        """
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
        """
        _generator = lambda a, b, x: (1 / (b**a * ss.gamma(a))) * np.power(
            x, a - 1) * np.exp(-x / b)
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.x)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        # there is no apparent explanation for reversing gammainc's parameter, but it works quite perfectly in my prototype
        _generator = lambda a, b, x: 1 - ss.gammainc(a, x / b)  
        
        if plot == True:
            x = np.linspace(-interval, interval, threshold)
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.x)    

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gamma distribution evaluated at some random variable.
        """
        if x_lower < 0:
            raise Exception('x_lower cannot be lower than 0. Entered value: {}'.format(x_lower))
        if x_upper is None:
            x_upper = self.x
        _cdf_def = lambda a, b, x: 1 - ss.gammainc(a, x / b) 

        return _cdf_def(self.a, self.b, x_upper, self.lambda_) - _cdf_def(self.a, self.b, x_lower, self.lambda_)

    def mean(self):
        """
        Returns: Mean of the Gamma distribution
        """
        return self.a * self.b

    def median(self):
        """
        Returns: Median of the Gamma distribution. 
        """
        return "No simple closed form."

    def mode(self):
        """
        Returns: Mode of the Gamma distribution
        """
        return (self.a - 1) * self.b

    def var(self):
        """
        Returns: Variance of the Gamma distribution
        """
        return self.a * pow(self.b,2)
    
    def std(self):
        """
        Returns: Standard deviation of the Gamma distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Gamma distribution
        """
        return 2 / np.sqrt(self.a)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Gamma distribution
        """
        return 6 / self.a

    def entropy(self):
        """
        Returns: differential entropy of the Gamma distribution

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k = self.a; theta = self.b
        return k +np.log(theta)+np.log(ss.gamma(k))-(1-k)*ss.digamma(k)

    def summary(self):
        """
        Returns: summary statistic regarding the Gamma distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# semi-infinite
class Pareto(Base):
    """
    This class contains methods concerning the Pareto Distribution Type 1. 

    Args:

        scale(float | x>0): scale parameter.
        shape(float | x>0): shape parameter.
        x(float | [shape, infty]): random variable.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Barry C. Arnold (1983). Pareto Distributions. International Co-operative Publishing House. ISBN 978-0-89974-012-6.
    - Wikipedia contributors. (2020, December 1). Pareto distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:00, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Pareto_distribution&oldid=991727349
    """
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
        """
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
        """
        x_m = self.scale
        alpha = self.shape

        def _generator(x, x_m, alpha):
            if x >= x_m:
                return (alpha * pow(x_m,alpha)) / np.power(x, alpha + 1)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, x_m, alpha) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.x, x_m, alpha)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        x_m = self.scale
        alpha = self.shape

        def _generator(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i, x_m, alpha) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.x, x_m, alpha)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Pareto distribution evaluated at some random variable.
        """
        if x_lower < 0:
            x_lower = 0
        if x_upper is None:
            x_upper = self.x

        def _cdf_def(x, x_m, alpha):
            if x >= x_m:
                return 1 - np.power(x_m / x, alpha)
            return 0
        return _cdf_def(x_upper, self.scale, self.alpha)+_cdf_def(x_lower, self.scale, self.alpha)

    def mean(self):
        """
        Returns: Mean of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale

        if a <= 1:
            return np.inf
        return (a * x_m) / (a - 1)

    def median(self):
        """
        Returns: Median of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        return x_m * pow(2, 1 / a)

    def mode(self):
        """
        Returns: Mode of the Pareto distribution.
        """
        return self.scale

    def var(self):
        """
        Returns: Variance of the Pareto distribution.
        """
        a = self.shape
        x_m = self.scale
        if a <= 2:
            return np.inf
        return (pow(x_m,2) * a) / (pow(a - 1,2) * (a - 2))

    def std(self):
        """
        Returns: Variance of the Pareto distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Pareto distribution. 
        """
        a = self.shape
        x_m = self.scale
        if a > 3:
            scale = (2 * (1 + a)) / (a - 3)
            return scale * sqrt((a - 2) / a)
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Pareto distribution. 
        """
        a = self.shape
        x_m = self.scale
        if a > 4:
            return (6 * (a**3 + a**2 - 6 * a - 2)) / (a * (a - 3) * (a - 4))
        return "undefined"

    def entropy(self):
        """
        Returns: differential entropy of the Pareto distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a = self.shape
        x_m = self.scale
        return np.log(x_m/a)+1+(1/a)

    def summary(self):
        """
        Returns: summary statistic regarding the Pareto distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# resolve pvalue
class Log_normal(Base):
    """
    This class contains methods concerning the Log Normal Distribution. 

    Args:
        
        randvar(float | [0, infty)): random variable
        mean_val(float): mean parameter
        std_val(float | x>0): standard deviation

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    References:
    - Weisstein, Eric W. "Log Normal Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/LogNormalDistribution.html
    - Wikipedia contributors. (2020, December 18). Log-normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:49, December 23, 2020, from https://en.wikipedia.org/w/index.php?title=Log-normal_distribution&oldid=994919804
    """
    def __init__(self, randvar, mean, std_val):
        if randvar<0:
            raise Exception('random variable should be greater than 0. Entered value for randvar:{}'.format(randvar))
        if std<0:
            raise Exception('random variable should be greater than 0. Entered value for std:{}'.format(std))
        self.randvar = randvar
        self.mean_val = mean_val
        self.std_val = std_val

    def pdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda mean, std, x: (1 / (x * std * np.sqrt(
            2 * np.pi))) * np.exp(-(np.log(x - mean)**2) / (2 * std**2))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.mean_val, self.std_val, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.mean_val, self.std_val, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda mean, std, x:0.5+ 0.5*ss.erfc(-(np.log(x - mean) /
                                                           (std * np.sqrt(2))))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.mean_val, self.std_val, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.mean_val, self.std_val, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Log Normal-distribution evaluated at some random variable.
        """
        _cdf_def = lambda mean, std, x:0.5+ 0.5*ss.erfc(-(np.log(x - mean) /
                                                           (std * np.sqrt(2))))
        if x_lower <0:
            raise Exception('x_lower should not be less then 0. X_lower: {}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar

        return _cdf_def(self.mean_val, self.std_val, x_upper)-_cdf_def(self.mean_val, self.std_val, x_lower)

    def mean(self):
        """
        Returns: Mean of the log normal distribution.
        """
        return np.exp(self.mean_val + pow(self.std_val,2) / 2)

    def median(self):
        """
        Returns: Median of the log normal distribution.
        """
        return np.exp(self.mean_val)

    def mode(self):
        """
        Returns: Mode of the log normal distribution.
        """
        return np.exp(self.mean_val - pow(self.std_val,2))

    def var(self):
        """
        Returns: Variance of the log normal distribution.
        """
        std = self.std_val
        mean = self.mean_val
        return (np.exp(pow(std,2)) - 1) * np.exp(2 * mean + pow(std,2))
    
    def std(self):
        """
        Returns: Standard deviation of the log normal distribution
        """
        return self.std_val

    def skewness(self):
        """
        Returns: Skewness of the log normal distribution.
        """
        std = self.std_val
        mean = self.mean_val
        return (np.exp(pow(std,2)) + 2) * np.sqrt(np.exp(pow(std,2)) - 1)

    def kurtosis(self):
        """
        Returns: Kurtosis of the log normal distribution.
        """
        std = self.std_val
        return np.exp(
            4 * pow(std,2)) + 2 * np.exp(3 * pow(std,2)) + 3 * np.exp(2 * pow(std,2)) - 6

    def entropy(self):
        """
        Returns: differential entropy of the log normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return self.mean_val+0.5*np.log(2*np.pi*np.e*self.std_val**2)

    def summary(self):
        """
        Returns: summary statistic regarding the log normal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# add pvalue method, check on ipynb
class Laplace(Base):
    """
    This class contains methods concerning Laplace Distirbution. 
    Args:
    
        location(float): mean parameter
        scale(float| x>0): standard deviation
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
        - Wikipedia contributors. (2020, December 21). Laplace distribution. In Wikipedia, The Free Encyclopedia. 
        Retrieved 10:53, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Laplace_distribution&oldid=995563221
    """
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
        """
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
        """
        _generator = lambda mu, b, x: (1 / (2 * b)) * np.exp(abs(x - mu) / b)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda mu, b, x: 1 / 2 + ((1 / 2) * np.sign(x - mu) *
                                              (1 - np.exp(abs(x - mu) / b)))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Laplace distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        _cdf_def = lambda mu, b, x: 1 / 2 + ((1 / 2) * np.sign(x - mu) * (1 - np.exp(abs(x - mu) / b)))
        
        return _cdf_def(self.location, self.scale, x_upper)-_cdf_def(self.location, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Laplace distribution.
        """
        return self.location

    def median(self):
        """
        Returns: Median of the Laplace distribution.
        """
        return self.location

    def mode(self):
        """
        Returns: Mode of the Laplace distribution.
        """
        return self.location

    def var(self):
        """
        Returns: Variance of the Laplace distribution.
        """
        return 2 * pow(self.scale, 2)

    def std(self):
        """
        Returns: Standard deviation of the Laplace distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Laplace distribution.
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Laplace distribution.
        """
        return 3

    def entropy(self):
        """
        Returns: differential entropy of the Laplace distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+np.log(2*self.scale)

    def summary(self):
        """
        Returns: summary statistic regarding the Laplace distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Logistic(Base):
    """
    This class contains methods concerning Logistic Distirbution. 
    Args:
    
        location(float): mean parameter
        scale(float | x>0): standard deviation
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 12). Logistic distribution. In Wikipedia, The Free Encyclopedia.
     Retrieved 11:14, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Logistic_distribution&oldid=993793195
    """
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
        """
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
        """
        _generator = lambda mu, s, x: np.exp(-(x - mu) / s) / (s * (1 + np.exp(
            -(x - mu) / s))**2)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda mu, s, x: 1 / (1 + np.exp(-(x - mu) / s))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def pvalue(self, x_lower=-np.inf, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logistic distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        _cdf_def = lambda mu, s, x: 1 / (1 + np.exp(-(x - mu) / s))
        return _cdf_def(self.location, self.scale, x_upper)- _cdf_def(self.location, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Logistic distribution.
        """
        return self.location

    def median(self):
        """
        Returns: Median of the Logistic distribution.
        """
        return self.location

    def mode(self):
        """
        Returns: Mode of the Logistic distribution.
        """
        return self.location

    def var(self):
        """
        Returns: Variance of the Logistic distribution.
        """
        return pow(self.scale,2) * pow(np.pi,2)/3

    def std(self):
        """
        Returns: Standard deviation of the Logistic distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Logistic distribution.
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Logistic distribution.
        """
        return 6 / 5

        def entropy(self):
        """
        Returns: differential entropy of the Logistic distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 2

    def summary(self):
        """
        Retruns: summary statistics of the Logistic distribution.
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Logit_normal(Base):
    """
    This class contains methods concerning Logit Normal Distirbution. 
    Args:
    
        sq_scale (float): squared scale parameter
        location(float): location parameter
        randvar(float | [0,1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 9). Logit-normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 07:44, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Logit-normal_distribution&oldid=993237113
    """
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
        """
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
        """
        _generator = lambda mu, sig, x: (1/(sig*np.sqrt(2*np.pi)))*np.exp(-((ss.logit(x)-mu)**2/(2*sig**2)))*(1/(x*(1-x)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.sq_scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.sq_scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda mu, sig, x: 1/2*(1+ss.erf((ss.logit(x)-mu)/(np.sqrt(2*sig**2))))
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.sq_scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.sq_scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logit distribution evaluated at some random variable.
        """
        if x_lower<0:
            raise Exception('x_lower should be a positive number. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        _cdf_def = lambda mu, sig, x: 1/2*(1+ss.erf((ss.logit(x)-mu)/(np.sqrt(2*sig**2))))
        return _cdf_def(self.location, self.sq_scale, x_upper)-_cdf_def(self.location, self.sq_scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Logit Normal distribution.
        """
        return "no analytical solution"


    def mode(self):
        """
        Returns: Mode of the Logit Normal distribution.
        """
        return "no analytical solution"

    def var(self):
        """
        Returns: Variance of the Logit Normal distribution.
        """
        return "no analytical solution"

    def std(self):
        """
        Returns: Standard deviation of the Logit Normal distribution.
        """
        return "no analytical solution"

    def entropy(self):
        """
        Returns: differential entropy of Logit Normal distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "unsupported"

    def summary(self):
        """
        Returns: Summary statistic regarding the Logit Normal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# semi-infinite
class Weibull(Base):
    """
    This class contains methods concerning Weibull Distirbution. Also known as Fréchet distribution.
    Args:
    
        shape(float | [0, infty)): mean parameter
        scale(float | [0, infty)): standard deviation
        randvar(float | [0, infty)): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 13). Weibull distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 11:32, December 28, 2020, from https://en.wikipedia.org/w/index.php?title=Weibull_distribution&oldid=993879185
    """
    def __init__(self, shape, scale, randvar=0.5):
        if shape<0 or scale<0 or randvar<0:
            raise Exception('all parameters should be a positive number. Entered values: shape: {0}, scale{1}, randvar{2}'.format(shape, scale, randvar))
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
        """
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
        """
        def _generator(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return (k/lambda_)*(x/lambda_)**(k-1)*np.exp(-(x/lambda_)**k)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        def _generator(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return 1-np.exp(-pow(x/lambda_, k))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Weilbull distribution evaluated at some random variable.
        """
        if x_lower<0:
            raise Exception('x_lower should be a positive number. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        def _cdf_def(_lamnda, k, x):
            if x<0:
                return 0
            if x>=0:
                return 1-np.exp(-pow(x/lambda_, k))

        return _cdf_def(self.location, self.shape, x_upper)-_cdf_def(self.location, self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Weibull distribution.
        """
        return self.scale*ss.gamma(1+(1/self.shape)

    def median(self):
        """
        Returns: Median of the Weibull distribution.
        """
        return self.scale*np.power(np.log(2), 1/self.shape)

    def mode(self):
        """
        Returns: Mode of the Weibull distribution.
        """
        k = self.shape
        if k>1:
            return self.scale*np.power((k-1)/k, 1/k)
        return 0

    def var(self):
        """
        Returns: Variance of the Weibull distribution.
        """
        lambda_ = self.scale
        k = self.shape
        return lambda_**2*(((ss.gamma(1+2/k)- ss.gamma(1+1/k)))**2)

    def std(self):
        """
        Returns: Standard deviation of the Weilbull distribution
        """
        return sqrt(self.var())

    def entropy(self):
        """
        Returns: differential entropy of the Weilbull distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        lambda_ = self.shape
        k = self.scale
        return (k+1)*np.euler_gamma/k+np.log(lambda_/k)+1

    def summary(self):
        """
        Returns: summary statistics of the Weilbull distribution.
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Weilbull_inv(Base):
    """
    This class contains methods concerning inverse Weilbull or the Fréchet Distirbution. 
    Args:
    
        shape(float | [0, infty)): shape parameter
        scale(float | [0,infty)]): scale parameter
        location(float | (-infty, infty)): location parameter
        randvar(float | randvar > location): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 7). Fréchet distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 07:28, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Fr%C3%A9chet_distribution&oldid=992938143
    """
    def __init__(self,  shape, scale, location, randvar):
        if shape<0 or scale<0:
            raise Exception('the value of scale and shape should be greater than 0. Entered values scale was:{0}, shape:{1}'.format(scale, shape))
        if randvar<location:
            raise Exception('random variable should be greater than the location parameter. Entered values: randvar: {0}, location:{1}'.format(randvar, location))
        self.shape = shape
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
        """
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
        """
        _generator = lambda a,s,m,x: (a/s)*np.power((x-m)/s, -1-a)*np.exp(-((x-m)/s)**-a)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.scale, self.location, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator =  lambda a,s,m,x: np.exp(-((x-m)/s)**-a)
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.shape, self.scale, self.location, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.scale, self.location, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Logit distribution evaluated at some random variable.
        """
        if x_lower<0:
            raise Exception('x_lower should be a positive number. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        _cdf_def = lambda a,s,m,x: np.exp(-((x-m)/s)**-a)
        return _cdf_def(self.shape, self.scale, self.location, x_upper)-_cdf_def(self.shape, self.scale, self.location, x_lower)

    def mean(self):
        """
        Returns: Mean of the Fréchet distribution.
        """
        if self.shape>1:
            return self.location + (self.scale*ss.gamma(1-1/self.shape))
        return np.inf

    def median(self):
        """
        Returns: Median of the Fréchet distribution.
        """
        return self.location + self.scale/(pow(np.log(2), 1/self.shape))

    def mode(self):
        """
        Returns: Mode of the Fréchet distribution.
        """
        return self.location + self.scale*(self.shape/pow(1+self.shape,1/self.shape))

    def var(self):
        """
        Returns: Variance of the Fréchet distribution.
        """
        a = self.shape
        s = self.scale
        if a>2:
            return pow(s,2)*(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a),2))
        return "infinity"

    def std(self):
        """
        Returns: Standard devtiation of the Fréchet distribution.
        """
        if self.var()=="infinity":
            return "infinity"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Fréchet distribution. 
        """
        a = self.shape
        if a>3:
            return (ss.gamma(1-3/a)-3*ss.gamma(1-2/a)*ss.gamma(1-1/a)+2*ss.gamma(1-1/a)**3)/pow(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a),2),3/2)
        return "infinity"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Fréchet distribution. 
        """
        a = self.shape
        if a>4:
            return -6+(ss.gamma(1-4/a)-4*ss.gamma(1-3/a)*ss.gamma(1-1/a)+3*pow(ss.gamma(1-2/a),2))/pow(ss.gamma(1-2/a)-pow(ss.gamma(1-1/a),2),2)
        return "infinity"

    def summary(self):
        """
        Returns: Summary statistic regarding the Fréchet distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Gumbel(Base):
    """
    This class contains methods concerning Gumbel Distirbution. 
    Args:
    
        location(float): location parameter
        scale(float>0): scale parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 26). Gumbel distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:22, December 29, 2020, from https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    """
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
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Gumbel distribution.
        """
        def _generator(mu, beta, x):
            z = (x-mu)/beta
            return (1/beta)*np.exp(-(z+np.exp(-z)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Gumbel distribution.
        """
        def _generator(mu, beta, x):
            return np.exp(-np.exp(-(x-mu)/beta))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def pvalue(self):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gumbel distribution evaluated at some random variable.
        """
        return "currently unsupported"

    def mean(self):
        """
        Returns: Mean of the Gumbel distribution.
        """
        return self.location+(self.scale*np.euler_gamma)

    def median(self):
        """
        Returns: Median of the Gumbel distribution.
        """
        return self.location - (self.scale*np.log(np.log(2)))

    def mode(self):
        """
        Returns: Mode of the Gumbel distribution.
        """
        return self.location

    def var(self):
        """
        Returns: Variance of the Gumbel distribution.
        """
        return (np.pi**2/6)*pow(self.scale,2)

    def std(self):
        """
        Returns: Standard deviation of the Gumbel distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Gumbel distribution. 
        """
        return 1.14

    def kurtosis(self):
        """
        Returns: Kurtosis of the Gumbel distribution. 
        """
        return 12/5

    def summary(self):
        """
        Returns: Summary statistic regarding the Gumbel distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Arcsine(Base):
    """
    This class contains methods concerning Arcsine Distirbution. 
    Args:

        randvar(float in [0, 1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.  

    Reference:
    - Wikipedia contributors. (2020, October 30). Arcsine distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:19, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Arcsine_distribution&oldid=986131091
    """
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
        """
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
        """
        _generator = lambda x: 1/(np.pi*np.sqrt(x*(1-x)))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        _generator = lambda x: (2/np.pi)*np.arcsin(np.sqrt(x))
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.location, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.location, self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Arcsine distribution evaluated at some random variable.
        """
        if x_lower<0 or x_lower>1:
            raise Exception('x_lower should only be in between 0 and 1. X_lower:{}'.format(x_lower))
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        _cdf_def = lambda x: (2/np.pi)*np.arcsin(np.sqrt(x))
        return _cdf_def(self.location, self.scale, x_upper)-_cdf_def(self.location, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Arcsine distribution.
        """
        return 1/2

    def median(self):
        """
        Returns: Median of the Arcsine distribution.
        """
        return 1/2

    def mode(self):
        """
        Returns: Mode of the Arcsine distribution. Mode is within the set {0,1}
        """
        return {0,1}

    def var(self):
        """
        Returns: Variance of the Arcsine distribution.
        """
        return 1/8

    def std(self):
        """
        Returns: Standard deviation of the Arcsine distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Arcsine distribution. 
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Arcsine distribution. 
        """
        return 3/2

    def summary(self):
        """
        Returns: Summary statistic regarding the Arcsine distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Triangular(Base):
    """
    This class contains methods concerning Triangular Distirbution. 
    Args:
    
        a(float): lower limit
        b(float | a<b): upper limit
        c(float| a≤c≤b): mode
        randvar(float | a≤randvar≤b): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.  

    Reference:
    - Wikipedia contributors. (2020, December 19). Triangular distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 05:41, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Triangular_distribution&oldid=995101682
    """
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
        """
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
        """
        def _generator(a,b,c,x):
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
            y = np.array([_generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.c, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        def _generator(a,b,c,x):
            if x<=a:
                return 0
            if a<x and x<=c:
                return pow(x-a,2)/((b-a)*(c-a))
            if c<x and x<b:
                return 1 - pow(b-x,2)/((b-c)*(b-c))
            if b<=x:
                return 1
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.c, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Triangular distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        def _cdf_def(a,b,c,x):
            if x<=a:
                return 0
            if a<x and x<=c:
                return pow(x-a,2)/((b-a)*(c-a))
            if c<x and x<b:
                return 1 - pow(b-x,2)/((b-c)*(b-c))
            if b<=x:
                return 1
        return _cdf_def(self.a, self.b, self.c, x_upper)-_cdf_def(self.a, self.b, self.c, x_lower)

    def mean(self):
        """
        Returns: Mean of the Triangular distribution.
        """
        return (self.a+self.b+self.c)/3

    def median(self):
        """
        Returns: Median of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        if c >= (a+b)/2:
            return a + sqrt(((b-a)*(c-a))/2)
        if c <= (a+b)/2:
            return b + sqrt((b-a)*(b-c)/2)

    def mode(self):
        """
        Returns: Mode of the Triangular distribution.
        """
        return self.c

    def var(self):
        """
        Returns: Variance of the Triangular distribution.
        """
        a = self.a
        b = self.b
        c = self.c
        return (1/18)*(pow(a,2)+pow(b,2)+pow(c,2)-a*b-a*c-b*c)

    def std(self):
        """
        Returns: Standard deviation of the Triangular distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Triangular distribution. 
        """
        a = self.a
        b = self.b
        c = self.c
        return (sqrt(2)*(a+b-2*c)*((2*a-b-c)*(a-2*b+c)))/(5*pow(a**2+b**2+c**2-a*b-a*c-b*c, 3/2))

    def kurtosis(self):
        """
        Returns: Kurtosis of the Triangular distribution. 
        """
        return -3/5

    def entropy(self):
        """
        Returns: differential entropy of the Triangular distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 0.5+np.log((self.b-self.a)*0.5)

    def summary(self):
        """
        Returns: Summary statistic regarding the Triangular distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Trapezoidal(Base):
    """
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
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
    """
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
        """
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
        """
        def _generator(a,b,c,d,x):
            if a<=x and x<b:
                return (2/(d+c-a-b))*(x-a)/(b-a)
            if b<=x and x<c:
                return (2/(d+c-a-b))
            if c<=x and x<=d:
                return (2/(d+c-a-b))*(d-x)/(d-c)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, self.c, self.d, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.c, self.d, self.randvar)

    def cdf(self,
            plot=False,
            interval=1,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
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
        """
        def _generator(a,b,c,d,x):
            if a<=x and x<b:
                return (x-a)**2/((b-a)*(d+c-a-b))
            if b<=x and x<c:
                return (2*x-a-b)/(d+c-a-b)
            if c<=x and x<=d:
                return 1- (d-x)**2/((d+c-a-b)*(d-c))
                
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, self.c, self.d, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.c, self.d, self.randvar)

    def pvalue(self):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Trapezoidal distribution evaluated at some random variable.
        """
        return "currently unsupported"

    def mean(self):
        """
        Returns: Mean of the Trapezoidal distribution.
        """
        return (self.a+self.b+self.c)/3


    # def var(self):
    #     """
    #     Returns: Variance of the Trapezoidal distribution. Currently Unsupported. 
    #     """
    #     return "currently unssuported."

    # def skewness(self):
    #     """
    #     Returns: Skewness of the Trapezoidal distribution. Currently Unsupported.
    #     """
    #     return "currently unssuported."

    # def kurtosis(self):
    #     """
    #     Returns: Kurtosis of the Trapezoidal distribution. Currently Unsupported.
    #     """
    #     return "currently unssuported."

    def summary(self):
        """
        Returns: Summary statistic regarding the Trapezoidal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class ARGUS(Base):
#     """
#     This class contains methods concerning ARGUS Distirbution. 
#     Args:
    
#         a(float|a<b): supported parameters
#         b(float): supported parameters
#         randvar(float | [a,b]): random variable

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2020, April 11). Trapezoidal distribution. In Wikipedia, The Free Encyclopedia. 
#     Retrieved 06:06, December 30, 2020, from https://en.wikipedia.org/w/index.php?title=Trapezoidal_distribution&oldid=950241388
#     """
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
#         """
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
#         """
#         def _generator(a,b,c,d,x):
#             if a<=x and x<b:
#                 return (2/(d+c-a-b))*(x-a)/(b-a)
#             if b<=x and x<c:
#                 return (2/(d+c-a-b))
#             if c<=x and x<=d:
#                 return (2/(d+c-a-b))*(d-x)/(d-c)

#         if plot == True:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.a, self.b, self.c, self.d, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.a, self.b, self.c, self.d, self.randvar)

#     def cdf(self,
#             plot=False,
#             interval=1,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
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
#         """
#         def _generator(a,b,c,d,x):
#             if a<=x and x<b:
#                 return (x-a)**2/((b-a)*(d+c-a-b))
#             if b<=x and x<c:
#                 return (2*x-a-b)/(d+c-a-b)
#             if c<=x and x<=d:
#                 return 1- (d-x)**2/((d+c-a-b)*(d-c))
                
#         if plot == True:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.a, self.b, self.c, self.d, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.a, self.b, self.c, self.d, self.randvar)

#     def pvalue(self):
#         """
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the Pareto distribution evaluated at some random variable.
#         """
#         return "currently unsupported"

#     def mean(self):
#         """
#         Returns: Mean of the ARGUS distribution.
#         """
#         return (self.a+self.b+self.c)/3

#     def median(self):
#         """
#         Returns: Median of the ARGUS distribution. Currently Unsupported.
#         """
#         return "currently unssuported."

#     def mode(self):
#         """
#         Returns: Mode of the ARGUS distribution. Currently Unsupported.
#         """
#         return "currently unssuported."

#     def var(self):
#         """
#         Returns: Variance of the ARGUS distribution. Currently Unsupported. 
#         """
#         return "currently unssuported."

#     def skewness(self):
#         """
#         Returns: Skewness of the ARGUS distribution. 
#         """
#         # return 

#     def kurtosis(self):
#         """
#         Returns: Kurtosis of the ARGUS distribution. Currently Unsupported.
#         """
#         return "currently unssuported."

#     def summary(self):
#         """
#         Returns: Summary statistic regarding the ARGUS distribution
#         """
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = " summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# to be reviewed: add variance and std functions
class Beta(Base):
    """
    This class contains methods concerning Beta Distirbution. 
    Args:
    
        alpha(float | x>0): shape
        beta(float | x>0): shape
        randvar(float | [0,1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 8). Beta distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 07:21, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_distribution&oldid=999043368
    """
    def __init__(self, alpha, beta, randvar):
        if randvar<0 | randvar>1:
            raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
        if alpha<0:
            raise ValueError('alpha parameter(shape) should be a positive number. Entered value:{}'.format(alpha))
        if beta<0:
            raise ValueError('beta parameter(shape) should be a positive number. Entered value:{}'.format(beta))

        self.alpha = alpha
        self.beta = beta 
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Beta distribution.
        """
        _generator = lambda a,b,x: (np.power(x,a-1)*np.power(1-x, b-1))/ss.beta(a,b)

        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Beta distribution.
        """
        _generator = lambda a,b,x: ss.betainc(a,b,x)
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.a, self.b, self.c, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.c, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Beta distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: ss.betainc(a,b,x)
        return _cdf_def(self.alpha, self.beta, x_upper)-_cdf_def(self.alpha, self.beta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Beta distribution.
        """
        return "currently unsupported."

    def median(self):
        """
        Returns: Median of the Beta distribution.
        """
        # warning: not yet validated.
        return ss.betainc(self.alpha, self.beta, 0.5)

    def mode(self):
        """
        Returns: Mode of the Beta distribution.
        """
        return "currently unsupported"

    def var(self):
        """
        Returns: Variance of the Beta distribution.
        """
        return "currently unsupported"

    def skewness(self):
        """
        Returns: Skewness of the Beta distribution. 
        """
        alpha = self.alpha; beta = self.beta
        return (2*(beta-alpha)*sqrt(alpha+beta+1))/((alpha+beta+2)*sqrt(alpha*beta))

    def kurtosis(self):
        """
        Returns: Kurtosis of the Beta distribution. 
        """
        alpha = self.alpha; beta = self.beta
        temp_up = 6*((alpha-beta)**2*(alpha+beta+1)-alpha*beta*(alpha+beta+2))
        return temp_up/(alpha*beta*(alpha+beta+2)*(alpha+beta+3))

    def entropy(self):
        """
        Returns: differential entropy of the Beta distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        alpha = self.alpha; beta = self.beta 
        return np.log(ss.beta(alpha, beta))-(alpha-1)*(ss.digamma(alpha)-ss.digamma(alpha+beta))-(beta-1)*(ss.digamma(beta)-ss.digamma(alpha+beta))

    def summary(self):
        """
        Returns: Summary statistic regarding the Beta distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# to be reviewed: add var and std functions
class Beta_prime(Base):
    """
    This class contains methods concerning Beta prime Distirbution. 
    Args:
    
        alpha(float | x>0): shape
        beta(float | x>0): shape
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, October 8). Beta prime distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:40, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_prime_distribution&oldid=982458594
    """
    def __init__(self, alpha, beta, randvar):
        if randvar<0:
            raise ValueError('random variable should not be less then 0. Entered value: {}'.format(randvar))
        if alpha<0:
            raise ValueError('alpha parameter(shape) should be a positive number. Entered value:{}'.format(alpha))
        if beta<0:
            raise ValueError('beta parameter(shape) should be a positive number. Entered value:{}'.format(beta))

        self.alpha = alpha
        self.beta = beta 
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval=0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Beta prime distribution.
        """
        _generator = lambda a,b,x: (np.power(x,a-1)*np.power(1+x, -a-b))/ss.beta(a,b)

        if plot == True:
            if interval<0:
                raise ValueError('random variable should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Beta prime distribution.
        """
        _generator = lambda a,b,x: ss.betainc(a,b,x/(1+x))
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Beta prime distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: ss.betainc(a,b,x/(1+x))
        return _cdf_def(self.alpha, self.beta, x_upper)-_cdf_def(self.alpha, self.beta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Beta prime distribution.
        """
        if self.beta>1:
            return self.alpha/(self.beta-1)
        return "currently unsupported."

    def median(self):
        """
        Returns: Median of the Beta prime distribution.
        """
        # warning: not yet validated.
        return "unsupported"

    def mode(self):
        """
        Returns: Mode of the Beta prime distribution.
        """
        if self.alpha>=1:
            return (self.alpha+1)/(self.beta+1)
        return 0

    def var(self):
        """
        Returns: Variance of the Beta prime distribution.
        """
        alpha = self.alpha
        beta = self.beta
        if beta>2:
            return (alpha*(alpha+beta-1))/((beta-2)*(beta-1)**2)
        return "currently unsupported"

    def std(self):
        """
        Returns: Standard deviation of the Log logistic distribution
        """
        if self.var() == "currently unsupported":
            return "currently unsupported"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Beta prime distribution. 
        """
        alpha = self.alpha; beta = self.beta
        if beta>3:
            scale = (2*(2*alpha+beta-1))/(beta-3)
            return scale*sqrt((beta-2)/(alpha*(alpha+beta-1)))
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Beta prime distribution. 
        """
        return "currently unsupported"

    def entropy(self):
        """
        Returns: differential entropy of the Beta prime distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return "currently unsupported"

    def summary(self):
        """
        Returns: Summary statistic regarding the Beta prime distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Bates(Base):
    """
    This class contains methods concerning Bates Distirbution. Also referred to as the regular mean distribution.

    Note that the Bates distribution is a probability distribution of the mean of a number of statistically indipendent uniformly
    distirbuted random variables on the unit interval. This is often confused with the Irwin-Hall distirbution which is 
    the distribution of the sum (not the mean) of n independent random variables. The two distributions are simply versions of 
    each other as they only differ in scale.
    Args:
    
        a(float): lower bound
        b(float |b>a): upper bound
        n(int | x>=1)
        randvar(float | [a,b]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 8). Bates distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 08:27, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Bates_distribution&oldid=999042206
    """
    def __init__(self, a, b, n, randvar):
        if randvar<0 | randvar>1:
            raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
        if a>b:
            raise ValueError('lower bound (a) should not be greater than upper bound (b).')
        if isinstance(n, int)==False:
            raise TypeError('parameter n should be an integer type.')

        self.a = a
        self.b = b 
        self.n = n
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Bates distribution.
        """
        def _generator(a,b,n, x):
            if a<x | x<b:
                bincoef = lambda n,k: np.math.factorial(n)/(np.math.factorial(k)*(np.math.factorial(n-k)))
                return np.sum([pow(-1,i)*bincoef(n,i)*np.power(((x-a)/(b-a)- i/n), n-1)*np.sign((x-a)/(b-1)-i/n) for i in range(0, n)])
            return 0

        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.a, self.b, self.n, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.n, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Bates distribution.
        """
        return "currently unsupported"

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Bates distribution evaluated at some random variable.
        """

        return "currently unsupported"

    def mean(self):
        """
        Returns: Mean of the Bates distribution.
        """
        return 0.5*(self.a+self.b)

    def var(self):
        """
        Returns: Variance of the Bates distribution.
        """
        return 1/(12*self.n)*pow(self.b-self.a,2)

    def std(self):
        """
        Returns: Standard devtiation of the Bates distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Bates distribution. 
        """
        return -6/(5*self.n)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Bates distribution. 
        """
        return 0

    def summary(self):
        """
        Returns: Summary statistic regarding the Bates distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# to be reviewed
class Erlang(Base):
    """
    This class contains methods concerning Erlang Distirbution. 
    Args:
    
        shape(int | x>0): shape
        rate(float | x>=0): rate
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 6). Erlang distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:38, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Erlang_distribution&oldid=998655107
    - Weisstein, Eric W. "Erlang Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/ErlangDistribution.html
    """
    def __init__(self, shape, rate, randvar):
        if randvar<0:
            raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
        if isinstance(shape,int)==False and shape>0:
            raise Exception('shape parameter should be an integer greater than 0.')
        if rate<0:
            raise ValueError('beta parameter(rate) should be a positive number. Entered value:{}'.formatrate))
        
        self.shape = shape
        self.rate = rate
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Erlang distribution.
        """
        _generator = lambda shape, rate, x: (np.power(rate, shape)*np.power(x,shape-1)*np.exp(-rate*x))/np.math.factorial((shape-1))

        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.shape, self.rate, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.rate, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Erlang distribution.
        """
        _generator = lambda shape, rate, x: ss.gammainc(shape, rate*x)/np.math.factorial(shape-1)
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.shape, self.rate, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.rate, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda shape, rate, x: ss.gammainc(shape, rate*x)/np.math.factorial(shape-1)
        return _cdf_def(self.shape, self.rate, x_upper)-_cdf_def(self.shape, self.rate, x_lower)

    def mean(self):
        """
        Returns: Mean of the Erlang distribution.
        """
        return self.shape/self.rate

    def median(self):
        """
        Returns: Median of the Erlang distribution.
        """
        return "no simple closed form"

    def mode(self):
        """
        Returns: Mode of the Erlang distribution.
        """
        return (1/self.rate)*(self.shape-1)

    def var(self):
        """
        Returns: Variance of the Erlang distribution.
        """
        return self.shape/pow(self.rate,2)

    def std(self):
        """
        Returns: Standard deviation of the Eerlang distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Erlang distribution. 
        """
        return 2/sqrt(self.shape)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Erlang distribution. 
        """
        return 6/self.shape

    def entropy(self):
        """
        Returns: differential entropy of the Erlang distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        k= self.shape; lambda_ = self.rate
        return (1-k)*ss.digamma(k)+np.log(ss.gamma(k)/lambda_)+k

    def summary(self):
        """
        Returns: Summary statistic regarding the Erlang distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# to be reviewed
class Maxwell_Boltzmann(Base):
    """
    This class contains methods concerning Maxwell-Boltzmann Distirbution. 
    Args:
    
        a(int | x>0): parameter
        randvar(float | x>=0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 12). Maxwell–Boltzmann distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 01:02, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Maxwell%E2%80%93Boltzmann_distribution&oldid=999883013
    """
    def __init__(self, a, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value: {}'.format(randvar))
        if a<0:
            raise ValueError('parameter a should be a positive number. Entered value:{}'.format(a))
        if isinstance(a,int) == False:
            raise TypeError('parameter should be in type int')

        self.a = a
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Maxwell-Boltzmann distribution.
        """
        _generator = lambda a, x: sqrt(2/np.pi)*(x**2*np.exp(-x**2/(2*a**2)))/(a**3)

        if plot == True:
            if interval<0:
                raise ValueError('interval should be a positive number. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Maxwell-Boltzmann distribution.
        """
        _generator = lambda a, x: ss.erf(x/(sqrt(2)*a))-sqrt(2/np.pi)*(x**2*np.exp(-x**2/(2*a**2)))/(a)
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Maxwell-Boltzmann distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a, x: ss.erf(x/(sqrt(2)*a))-sqrt(2/np.pi)*(x**2*np.exp(-x**2/(2*a**2)))/(a)
        return _cdf_def(self.a, x_upper)-_cdf_def(self.a, x_lower)

    def mean(self):
        """
        Returns: Mean of the Maxwell-Boltzmann distribution.
        """
        return 2*self.a*sqrt(2/np.pi)

    def median(self):
        """
        Returns: Median of the Maxwell-Boltzmann distribution.
        """
        return "currently unsupported"

    def mode(self):
        """
        Returns: Mode of the Maxwell-Boltzmann distribution.
        """
        return sqrt(2)*self.a

    def var(self):
        """
        Returns: Variance of the Maxwell-Boltzmann distribution.
        """
        return (self.a**2*(3*np.pi-8))/np.pi

    def std(self):
        """
        Returns: Standard deviation of the Maxwell-Boltzmann distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Maxwell-Boltzmann distribution. 
        """
        return (2*sqrt(2)*(16-5*np.pi))/np.power((3*np.pi-8), 3/2)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Maxwell-Boltzmann distribution. 
        """
        return 4*((-96+40*np.pi-3*np.pi**2)/(3*np.pi-8)**2)

    def entropy(self):
        """
        Returns: differential entropy of the Maxwell-Boltzmann distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        a= self.a
        return np.log(a*sqrt(2*np.pi)+np.euler_gamma-0.5)

    def summary(self):
        """
        Returns: Summary statistic regarding the Maxwell-Boltzmann distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Beta_rectangular(Base):
    """
    This class contains methods concerning Beta-rectangular Distirbution. 
    Thus it is a bounded distribution that allows for outliers to have a greater chance of occurring than does the beta distribution.

    Args:
    
        alpha(float): shape parameter
        beta (float): shape parameter
        theta(float | 0<x<1): mixture parameter
        min(float): lower bound
        max(float): upper bound
        randvar(float | alpha<=x<=beta): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution.  

    Reference:
    - Wikipedia contributors. (2020, December 7). Beta rectangular distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 01:05, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Beta_rectangular_distribution&oldid=992814814
    """
    def __init__(self, alpha, beta, theta, min, max, randvar):
        if alpha<0 or beta<0:
            raise ValueError('alpha and beta parameter should not be less that 0. Entered values: alpha: {}, beta: {}'.format(alpha, beta))
        if theta<0 or theta>1:
            raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(theta))
        if randvar<min and randvar>max: # should only return warning 
            raise ValueError('random variable should be between alpha and beta shape parameters. Entered value:{}'.format(randvar))
        
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.min = min
        self.max = max
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Beta-rectangular distribution.
        """
        def _generator(a,b, alpha, beta, theta, x):
            if x>a or x<b:
                return (theta*ss.gamma(alpha+beta)/(ss.gamma(alpha)*ss.gamma(beta))*(np.power(x-a, alpha-1)*np.power(b-x, beta-1))/(np.power(b-a, alpha+beta+1)))+(1-theta)/(b-a)
            return 0

        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.min, self.max, self.alpha, self.beta, self.theta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.min, self.max, self.alpha, self.beta, self.theta, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Beta-rectangular distribution.
        """
        def _generator(a,b,alpha, beta, theta, x):
            if x<=a:
                return 0
            elif x>a | x<b:
                z = (b)/(b-a)
                return theta*ss.betainc(alpha,beta,z)+((1-theta)*(x-a))/(b-a)
            else:
                return 1

        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.min, self.max, self.alpha, self.beta, self.theta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.min, self.max, self.alpha, self.beta, self.theta, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Beta-rectangular distribution evaluated at some random variable.
        """
        
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        def _cdf_def(a,b,alpha, beta, theta, x):
            if x<=a:
                return 0
            elif x>a or x<b:
                z = (b)/(b-a)
                return theta*ss.betainc(alpha,beta,z)+((1-theta)*(x-a))/(b-a)
            else:
                return 1

        return _cdf_def(self.min, self.max, self.alpha, self.beta, self.theta, x_upper)-_cdf_def(self.min, self.max, self.alpha, self.beta, self.theta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Beta-rectangular distribution.
        """
        alpha = self.alpha; beta =self.beta; theta=self.theta
        a = self.min; b=self.max
        return a+(b-a)*((theta*alpha)/(alpha+beta)+(1-theta)/2)

    def var(self):
        """
        Returns: Variance of the Beta-rectangular distribution.
        """
        alpha = self.alpha; beta =self.beta; theta=self.theta
        a = self.min; b=self.max
        k = alpha+beta
        return (b-a)**2*((theta*alpha*(alpha+1))/(k*(k+1))+(1-theta)/3-(k+theta*(alpha-beta))**2/(4*k**2))

    def std(self):
        """
        Returns: Standard deviation of the Beta-rectangular distribution.
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Beta-rectangular distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Bernoulli(Base):
    """
    This class contains methods concerning Continuous Bernoulli Distirbution. 
    The continuous Bernoulli distribution arises in deep learning and computer vision, 
    specifically in the context of variational autoencoders, for modeling the 
    pixel intensities of natural images

    Args:
    
        shape(float): parameter
        randvar(float | x in [0,1]): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 2). Continuous Bernoulli distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 02:37, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Continuous_Bernoulli_distribution&oldid=986761458
    - Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
    - Kingma, D. P., & Welling, M. (2014, April). Stochastic gradient VB and the variational auto-encoder. 
    In Second International Conference on Learning Representations, ICLR (Vol. 19).
    - Ganem, G & Cunningham, J.P. (2019). The continouous Bernoulli: fixing a pervasive error in variational autoencoders. https://arxiv.org/pdf/1907.06845.pdf
    """
    def __init__(self, shape, randvar):
        if randvar<0 or randvar>1:
            raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
        if shape<0 or shape>1:
            raise ValueError('shape parameter a should only be in between 0 and 1. Entered value:{}'.format(shape))
        
        self.shape = shape
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Continuous Bernoulli distribution.
        """
        C = lambda shape: (2*np.arctanh(1-2*shape))/(1-2*shape) if shape != 0.5 else 2
        def _generator(shape, x):
            return C(shape)*np.power(shape, x)*np.power(1-shape, 1-x)
            
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Continuous Bernoulli distribution.
        """
        _generator = lambda shape, x: (shape**x*np.power(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x

        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Continuous Bernoulli distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda shape, x: (shape**x*np.power(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x
        return _cdf_def(self.shape, x_upper)-_cdf_def(self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 0.5
        return shape/(2*shape-1)+(1/(2*np.arctanh(1-2*shape)))


    def var(self):
        """
        Returns: Variance of the Continuous Bernoulli distribution.
        """
        shape = self.shape
        if shape == 0.5:
            return 1/12
        return shape/((2*shape-1)**2)+1/(2*np.arctanh(1-2*shape))**2

    def std(self):
        """
        Returns: Standard deviation of the Continuous Bernoulli distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Continuous Bernoulli distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class Beta_noncentral(Base):
#     """
#     This class contains methods concerning noncentral beta Distirbution (type 1). 

#     Args:

#         alpha(float | x>0): shape parameter
#         beta(float | x>0): shape parameter
#         noncentrality(float | x>=0): noncentrality parameter
#         randvar(float | x in [0,1]): random variable

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - entropy for differential entropy of the distribution.
#         - summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2020, November 2). Continuous Bernoulli distribution. In Wikipedia, The Free Encyclopedia. 
#     Retrieved 02:37, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Continuous_Bernoulli_distribution&oldid=986761458
#     - Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
#     - Kingma, D. P., & Welling, M. (2014, April). Stochastic gradient VB and the variational auto-encoder. 
#     In Second International Conference on Learning Representations, ICLR (Vol. 19).
#     """
#     def __init__(self, alpha, beta, noncentral, randvar):
#         if randvar<0 or randvar>1:
#             raise ValueError('random variable should only be in between 0 and 1. Entered value: {}'.format(randvar))
#         if shape<0 or shape>1:
#             raise ValueError('shape parameter a should only be in between 0 and 1. Entered value:{}'.format(shape))
        
#         self.shape = shape
#         self.randvar = randvar

#     def pdf(self,
#             plot=False,
#             threshold=1000,
#             interval = 1,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either probability density evaluation for some point or plot of noncentral beta distribution.
#         """
        
#         _generator = lambda a,b, lambda_: np.sum([np.exp(-lambda_/2)])
            
#         if plot == True:
#             if interval<0:
#                 raise ValueError('random variable should not be less then 0. Entered value: {}'.format(interval))
#             x = np.linspace(0, 1, int(threshold))
#             y = np.array([_generator(self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.shape, self.randvar)

#     def cdf(self,
#             plot=False,
#             threshold=1000,
#             interval = 1,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either cumulative distribution evaluation for some point or plot of noncentral beta distribution.
#         """
#         _generator = lambda shape, x: (shape**x*np.power(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x

#         if plot == True:
#             if interval<0:
#                 raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
#             x = np.linspace(0, interval, int(threshold))
#             y = np.array([_generator(self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.shape, self.randvar)

#     def pvalue(self, x_lower=0, x_upper=None):
#         """
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the non-central distribution evaluated at some random variable.
#         """
#         if x_upper == None:
#             x_upper = self.randvar
#         if x_lower>x_upper:
#             raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
#         _cdf_def  = lambda shape, x: (shape**x*np.power(1-shape, 1-x)+shape-1)/(2*shape-1) if shape != 0.5 else x
#         return _cdf_def(self.shape, x_upper)-_cdf_def(self.shape, x_lower)

#     def mean(self):
#         """
#         Returns: Mean of the noncentral beta distribution.
#         """
#         shape = self.shape
#         if shape == 0.5:
#             return 0.5
#         return shape/(2*shape-1)+(1/(2*np.arctanh(1-2*shape)))


#     def var(self):
#         """
#         Returns: Variance of the noncentral beta distribution.
#         """
#         shape = self.shape
#         if shape == 0.5:
#             return 1/12
#         return shape/((2*shape-1)**2)+1/(2*np.arctanh(1-2*shape))**2

#     def summary(self):
#         """
#         Returns: Summary statistic regarding the noncentral beta distribution
#         """
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = " summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

class Wigner(Base):
    """
    This class contains methods concerning Wigner semicricle Distirbution. 
    Args:
    
        radius(int | x>0): parameter
        randvar(float | x in [-radius, radius]): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 14). Wigner semicircle distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 03:41, January 14, 2021, from https://en.wikipedia.org/w/index.php?title=Wigner_semicircle_distribution&oldid=994143777
    """
    def __init__(self, radius, randvar=0):
        if radius<0:
            raise ValueError('parameter a should be a positive number. Entered value:{}'.format(radius))
         if randvar<-radius or randvar>radius:
            raise ValueError('random variable should only be in between -radus and radius. Entered value: {}'.format(randvar))

        self.radius = radius
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Wigner semicricle distribution.
        """
        _generator = lambda r, x: 2/(np.pi*r**2)*sqrt(r**2-x**2)

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.radius, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.radius, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Wigner semicricle distribution.
        """
        _generator = lambda r,x: 0.5+(x*sqrt(r**2-x**2))/(np.pi*r**2)+(np.arcsin(x/r))/np.pi
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.radius, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.radius, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Wigner semicricle distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda r,x: 0.5+(x*sqrt(r**2-x**2))/(np.pi*r**2)+(np.arcsin(x/r))/np.pi
        return _cdf_def(self.radius, x_upper)-_cdf_def(self.radius, x_lower)

    def mean(self):
        """
        Returns: Mean of the Wigner semicricle distribution.
        """
        return 0

    def median(self):
        """
        Returns: Median of the Wigner semicricle distribution.
        """
        return 0

    def mode(self):
        """
        Returns: Mode of the Wigner semicricle distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Wigner semicricle distribution.
        """
        return self.radius**2/4
    
    def std(self):
        """
        Returns: Standard deviation of the Wigner semicircle distribution.
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Wigner semicricle distribution. 
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Wigner semicricle distribution. 
        """
        return -1

    def entropy(self):
        """
        Returns: differential entropy of the Wigner semicricle distribution.
        """
        return np.log(np.pi*self.raduis)-0.5

    def summary(self):
        """
        Returns: Summary statistic regarding the Wigner semicricle distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Balding_Nichols(Base):
    """
    This class contains methods concerning Balding Nichols Distirbution. 
    Args:
    
        F(float | 0<=x<=1): F parameter
        p(float | 0<=x<=1): p parameter
        randvar(float | 0<=x<=1): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 6). Balding–Nichols model. In Wikipedia, The Free Encyclopedia. 
    Retrieved 04:55, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Balding%E2%80%93Nichols_model&oldid=998640082
    """
    def __init__(self, F, p, randvar=0.5):
        if randvar<0 or randvar>1:
            raise ValueError('random variable should only be in between 0 and 1. Entered value:{}'.format(randvar))
         if p<0 or p>1:
            raise ValueError('parameter p should only be in between 0 and 1. Entered value: {}'.format(p))
         if F<0 or F>1:
            raise ValueError('parameter F should only be in between 0 and 1. Entered value: {}'.format(F))

        self.F = F
        self.p = p
        self.alpha = (1-F/F)*p
        self.beta = (1-F/F)*(1-p)
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Balding Nichols distribution.
        """
        _generator = lambda alpha, beta, x: (x**(alpha-1)*np.power(1-x, beta-1))/ss.beta(alpha, beta)
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Balding Nichols distribution.
        """
        _generator = lambda alpha, beta, x: ss.betainc(alpha, beta, x)
        if plot == True:
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Balding Nichols distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda alpha, beta, x: ss.betainc(alpha, beta, x)
        return _cdf_def(self.alpha, self.beta, x_upper)-_cdf_def(self.alpha, self.beta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Balding Nichols distribution.
        """
        return self.p

    def median(self):
        """
        Returns: Median of the Balding Nichols distribution.
        """
        return "no simple closed form"

    def mode(self):
        """
        Returns: Mode of the Balding Nichols distribution.
        """
        return (self.F-(1-self.F)*self.p)/(3*(self.F-1))

    def var(self):
        """
        Returns: Variance of the Balding Nichols distribution.
        """
        return self.F*self.p*(1-self.p)

    def std(self):
        """
        Returns: Standard deviation of the Balding Nichols distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Balding Nichols distribution. 
        """
        F = self.f
        p = self.p
        return (2*F*(1-2*p))/((1+F)*sqrt(F*(1-p)*p))

    def summary(self):
        """
        Returns: Summary statistic regarding the Balding Nichols distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class PERT(Base):
    
    pass
# semi-infinite
class Benini(Base):
    """
    This class contains methods concerning Benini Distirbution. 
    Args:
    
        alpha(float | x>0): shape parameter
        beta(float | x>0): shape parameter
        sigma(float | x>0): scale parameter
        randvar(float | x>sigma): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, August 10). Benini distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:00, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Benini_distribution&oldid=972072772
    - Mathematica (2021). BeniniDistribution. Retrieved from https://reference.wolfram.com/language/ref/BeniniDistribution.html
    """
    def __init__(self, alpha, beta, sigma, randvar):
        if randvar<sigma:
            raise ValueError('random variable should be less than sigma. radvar={}, sigma={}'.format(randvar, sigma))
         if alpha<0 or beta<0 or sigma<0:
            raise ValueError('shape and scale parameters should be a positive number. Entered value: {}'.format(alpha, beta, sigma))

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Benini distribution.
        """
        _generator = lambda a,b,o,x:np.exp(-a*np.log10(x/o)-b*(np.log10(x/o)**2))*(a/x+(2*b*np.log10(x/o))/x) if x>0 else 0
        if plot == True:
            if interval<0:
                raise ValueError('interval should be a positive number. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, self.sigma, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.sigma, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Benini distribution.
        """
        _generator = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*(np.log10(x/o))**2)
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.alpha, self.sigma, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.sigma, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Benini distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*(np.log10(x/o))**2) if x>0 else 0
        return _cdf_def(self.alpha, self.beta, self.sigma, x_upper)-_cdf_def(self.alpha, self.beta, self.sigma, x_lower)

    def mean(self):
        """
        Returns: Mean of the Benini distribution.
        """
        alpha = self.alpha; beta = self.beta; sigma = self.sigma
        return sigma+(sigma/(sqrt(2*beta)))*ss.hermite(-1, (1+alpha)/(sqrt(2*beta)))

    def median(self):
        """
        Returns: Median of the Benini distribution.
        """
        alpha = self.alpha; beta = self.beta; sigma = self.sigma
        return sigma*np.exp((-alpha+sqrt(alpha**2+beta*np.log(16)))/(2*beta))


    def var(self):
        """
        Returns: Variance of the Benini distribution.
        """
        mean = self.mean()
        alpha = self.alpha; beta = self.beta; sigma = self.sigma
        return (sigma**2+(2*sigma**2)/sqrt(2*beta)*ss.hermite(-1,(2+alpha)/sqrt(2*beta)))-mean**2

    def std(self):
        """
        Returns: Variance of the Exponential distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Benini distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Normal_folded(Base):
    """
    This class contains methods concerning Folded Normal Distirbution. 
    Args:
    
        loc(float): location parameter
        scale(float | x>0): scale parameter. Note that this is not the squared value of scale.
        randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, October 19). Folded normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:18, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Folded_normal_distribution&oldid=984315189
    - Tsagris, M.; Beneki, C.; Hassani, H. (2014). "On the folded normal distribution". Mathematics. 2 (1): 12–28. arXiv:1402.3559
    """
    def __init__(self, loc, scale, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value:{}'.format(randvar))
        if scale<0:
            raise ValueError('scale should be a positive number. Entered value:{}'.format(scale))
        
        self.loc = loc
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Folded Normal distribution.
        """
        _generator = lambda mu, sig, x: 1/(sig*sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*sig**2))+1/(sig*sqrt(2*np.pi))*np.exp(-(x+mu)**2/(2*sig**2)) if x<0 else 0
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Folded Normal distribution.
        """
        _generator = lambda mu, sig, x: 0.5*(ss.erf((x+mu)/(sig*sqrt(2)))+ss.erf((x-mu)/(sig*sqrt(2)))) if x>0 else 0
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale,  i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Folded Normal distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda mu, sig, x: 0.5*(ss.erf((x+mu)/(sig*sqrt(2)))+ss.erf((x-mu)/(sig*sqrt(2)))) if x>0 else 0
        return _cdf_def(self.loc, self.scale, x_upper)-_cdf_def(self.loc, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Folded Normal distribution.
        """
        return "currently unsupported."


    def var(self):
        """
        Returns: Variance of the Folded Normal distribution.
        """
        return "currently unsupported."

    def std(self):
        """
        Returns: Standard deviation of the Folded Normal distribution
        """
        return "currently unsupported"

    def summary(self):
        """
        Returns: Summary statistic regarding the Folded Normal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Logistic_half(Base):
    """
    This class contains methods concerning half logistic Distirbution. 
    Args:
    
        k(float | x>0): parameter
        randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2019, November 8). Half-logistic distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:27, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Half-logistic_distribution&oldid=925231753
    """
    def __init__(self, k, randvar=0.5):
        if randvar<0 or k<0:
            raise ValueError('random variable and parameter k be a positive number. Entered value: k = {}, randvar = {}'.format(k,randvar))
 
        self.k = k
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of half logistic distribution.
        """
        _generator = lambda k,x: 2*np.exp(-k)/(1+np.exp(-k))**2 if x>0 else 0
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.k i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.k self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of half logistic distribution.
        """
        _generator = lambda k,x: (1-np.exp(-k))/(1+np.exp(-k)) if x>0 else 0
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.k, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.k, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the half logistic distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda k,x: (1-np.exp(-k))/(1+np.exp(-k)) if x>0 else 0
        return _cdf_def(self.k, x_upper)-_cdf_def(self.k, x_lower)

    def mean(self):
        """
        Returns: Mean of the half logistic distribution.
        """
        return np.log(4)

    def median(self):
        """
        Returns: Median of the half logistic distribution.
        """
        return np.log(3)

    def mode(self):
        """
        Returns: Mode of the half logistic distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the half logistic distribution.
        """
        return np.pi**2/3-(np.log(4))**2

    def std(self):
        """
        Returns: Standard deviation of the half logistic distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the half logistic distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Normal_half(Base):
    """
    This class contains methods concerning Half Normal Distirbution. 
    Args:
    
        scale(float | x>0): scale parameter
        randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 30). Half-normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 06:42, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Half-normal_distribution&oldid=997191556
    - Weisstein, Eric W. "Half-Normal Distribution." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/Half-NormalDistribution.html
    """
    def __init__(self, scale, randvar=0.5):
        if scale<0 or randvar<0:
            raise ValueError('random variable and scale parameter should be a positive number. Entered value: scale = {}, randvar={}'.format(scale, randvar))

        self. scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Half Normal distribution.
        """
        _generator = lambda sig, x: sqrt(2)/(sig*sqrt(np.pi))*np.exp(-x**2/(2*sig**2)) if x>0 else 0
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Half Normal distribution.
        """
        _generator = lambda sig, x: ss.erf(x/(sig*sqrt(2))) if x>0 else 0
        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Helf Normal distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda sig, x: ss.erf(x/(sig*sqrt(2))) if x>0 else 0
        return _cdf_def(self.scale, x_upper)-_cdf_def(self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Half Normal distribution.
        """
        return self.scale*sqrt(2)/sqrt(np.pi)

    def median(self):
        """
        Returns: Median of the Half Normal distribution.
        """
        return self.scale*sqrt(2)*ss.erfinv(0.5)

    def mode(self):
        """
        Returns: Mode of the Half Normal distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Half Normal distribution.
        """
        return self.scale**2*(1-2/np.pi)

    def std(self):
        """
        Returns: Standard deviation of the Half Normal distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Half Normal distribution. 
        """
        return (sqrt(2)*(4-np.pi))/np.power((np.pi-2), 3/2)

    def kurtosis(self):
        """
        Returns: kurtosis of the Half Normal distribution
        """
        return (8*(np.pi-3))/(np.pi-2)**2
    
    def entropy(self):
        """
        Returns: differential entropy of the Half Normal distribution
        """
        return 0.5*np.log2(2*np.pi*np.e*self.scale**2)-1

    def summary(self):
        """
        Returns: Summary statistic regarding the Half Normal distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# asses cdf and pdf
class Gaussian_inv(Base):
    """
    This class contains methods concerning Inverse Gaussian Distirbution. 
    Args:
    
        mean(float | x>0): mean parameter
        scale(float | x>0): scale parameter
        randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 26). Inverse Gaussian distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:15, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Inverse_Gaussian_distribution&oldid=990830775
    - Weisstein, Eric W. "Inverse Gaussian Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/InverseGaussianDistribution.html
    """
    def __init__(self, mean_val, scale, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value:{}'.format(randvar))
         if scale<0 or mean<0:
            raise ValueError('mean and scale parameter should be a positive number. Entered value: mean = {}, scale = {}'.format(mean, scale))

        self.mean_val = mean_val
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Inverse Gaussian distribution.
        """
        _generator = lambda scale, mean, x: sqrt(scale/(2*np.pi*x**3))*np.exp(-(scale*(x-mean)**2)/(2*mean**2*x))
        if plot == True:
            if interval<0:
                raise ValueError('random variable should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, self.mean_val, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.mean_val, self.randvar)

    # def cdf(self,
    #         plot=False,
    #         threshold=1000,
    #         interval = 1,
    #         xlim=None,
    #         ylim=None,
    #         xlabel=None,
    #         ylabel=None):
    #     """
    #     Args:
        
    #         interval(int): defaults to none. Only necessary for defining plot.
    #         threshold(int): defaults to 1000. Defines the sample points in plot.
    #         plot(bool): if true, returns plot.
    #         xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
    #         ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
    #         xlabel(string): sets label in x axis. Only relevant when plot is true. 
    #         ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
    #     Returns: 
    #         either cumulative distribution evaluation for some point or plot of Inverse Gaussian distribution.
    #     """
    #     def _generator(mean, scale):
    #         normal_cdf = lambda mu, sig, x: 0.5*(1+ss.erf((x-mu)/(sig*sqrt(2))))
    #         normal_cdf()
    #     if plot == True:
    #         if interval<0:
    #             raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
    #         x = np.linspace(0, interval, int(threshold))
    #         y = np.array([_generator(self.alpha, self.beta, i) for i in x])
    #         return super().plot(x, y, xlim, ylim, xlabel, ylabel)
    #     return _generator(self.alpha, self.beta, self.randvar)

    # def pvalue(self, x_lower=0, x_upper=None):
    #     """
    #     Args:

    #         x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
    #         x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

    #         Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
    #         Otherwise, the default random variable is x.

    #     Returns:
    #         p-value of the Inverse Gaussian distribution evaluated at some random variable.
    #     """
    #     if x_upper == None:
    #         x_upper = self.randvar
    #     if x_lower>x_upper:
    #         raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
    #     _cdf_def  = lambda alpha, beta, x: ss.betainc(alpha, beta, x)
    #     return _cdf_def(self.alpha, self.beta, x_upper)-_cdf_def(self.alpha, self.beta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Inverse Gaussian distribution.
        """
        return self.mean


    def mode(self):
        """
        Returns: Mode of the Inverse Gaussian distribution.
        """
        mean = self.mean
        scale = self.scale
        return mean*(sqrt(1+(9*mean**2)/(4*scale**2))-(3*mean)/(2*scale))

    def var(self):
        """
        Returns: Variance of the Inverse Gaussian distribution.
        """
        return self.mean**3/self.scale

    def std(self):
        """
        Returns: Standard deviation of the Inverse Gaussian distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Inverse Gaussian distribution. 
        """
        return 3*sqrt(self.mean/self.scale)
    
    def kurtosis(self):
        """
        Returns: Kurtosis of the Inverse Gaussian distribution.
        """
        return (15*self.mean)/self.scale

    def summary(self):
        """
        Returns: Summary statistic regarding the Inverse Gaussian distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Gamma_inv(Base):
    """
    This class contains methods concerning Inverse Gamma Distirbution. 
    Args:
    
        alpha(float | x>0): shape parameter
        beta(float | x>0): scale parameter
        randvar(float | x>0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 26). Inverse-gamma distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:18, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Inverse-gamma_distribution&oldid=996362763
    - Mathematica (2021). InverseGammaDistribution. Retrieved from: https://reference.wolfram.com/language/ref/InverseGammaDistribution.html
    """
    def __init__(self, alpha, beta, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value:{}'.format(randvar))
         if alpha<0 or beta<0:
            raise ValueError('shape and scale parameter should be a positive number. Entered value: shape= {}, scale ={}'.format(alpha, beta))

        self.alpha = alpha
        self.beta = beta
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Inverse Gamma distribution.
        """
        _generator = lambda alpha, beta, x: (beta**alpha)/ss.gamma(alpha)*np.power(x, -alpha-1)*np.exp(-beta/x)
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Inverse Gamma distribution.
        """
        _generator = lambda alpha, beta, x: ss.gammainc(alpha, beta/x)/ss.gamma(alpha)
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.alpha, self.beta, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.alpha, self.beta, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Inverse Gamma distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda alpha, beta, x: ss.gammainc(alpha, beta/x)/ss.gamma(alpha) if x>0 else 0
        return _cdf_def(self.alpha, self.beta, x_upper)-_cdf_def(self.alpha, self.beta, x_lower)

    def mean(self):
        """
        Returns: Mean of the Inverse Gamma distribution.
        """
        if self.alpha>1:
            return self.beta/(self.alpha-1)
        return "undefined"

    def mode(self):
        """
        Returns: Mode of the Inverse Gamma distribution.
        """
        return self.beta/(self.alpha+1)

    def var(self):
        """
        Returns: Variance of the Inverse Gamma distribution.
        """
        if self.alpha>2:
            return self.beta**2/((self.alpha-1)**2*(self.alpha-2))
        return "undefined"

    def std(self):
        """
        Returns: Standard deviation of the Invese Gamma distribution
        """
        if self.var() == "undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Inverse Gamma distribution. 
        """
        if self.alpha>3:
            return (4*sqrt(self.alpha-2))/(self.alpha-3)
        return "undefined"

    def kurtosis(self):
        """
        Returns: kurtosis of the Inverse Gamma distribution
        """
        alpha = self.alpha
        if alpha >4:
            return 6*(5*alpha-11)/((alpha-3)*(alpha-4))
        return "undefined"
    
    def entropy(self):
        """
        Returns: differential entropy of the Inverse Gamma distribution
        """
        return self.alpha+np.log(self.beta*ss.gamma(self.alpha))-(1+self.alpha)*ss.digamma(self.alpha)

    def summary(self):
        """
        Returns: Summary statistic regarding the Inverse Gamma distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class Burr(Base):
#     """
#     This class contains methods concerning Burr Distirbution. 
#     Args:
    
#         alpha(float | x>0): shape parameter
#         beta(float | x>0): shape parameter
#         sigma(float | x>0): scale parameter
#         randvar(float | x>sigma): random variable

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - entropy for differential entropy of the distribution.
#         - summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2020, August 10). Benini distribution. In Wikipedia, The Free Encyclopedia. 
#     Retrieved 06:00, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Benini_distribution&oldid=972072772
#     - Mathematica (2021). BeniniDistribution. Retrieved from https://reference.wolfram.com/language/ref/BeniniDistribution.html
#     """
#     def __init__(self, alpha, beta, sigma, randvar):
#         if randvar<0:
#             raise ValueError('random variable shoould be a positive number. Entered value:{}'.format(randvar))
#          if alpha<0 or beta<0 or sigma<0:
#             raise ValueError('shape and scale parameters should be a positive number. Entered value: {}'.format(alpha, beta, sigma))

#         self.alpha = alpha
#         self.beta = beta
#         self.sigma = sigma
#         self.randvar = randvar

#     def pdf(self,
#             plot=False,
#             threshold=1000,
#             interval = 1,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either probability density evaluation for some point or plot of Burr distribution.
#         """
        # _generator = lambda a,b,o,x:np.exp(-a*np.log10(x/o)-b*pow(np.log10(x/o),2))*(a/x+(2*b*np.log10(x/o))/x) 
#         if plot == True:
#             if interval<0:
#                 raise ValueError('random variable should not be less then 0. Entered value: {}'.format(interval))
#             x = np.linspace(0, 1, int(threshold))
#             y = np.array([_generator(self.alpha, self.beta, self.sigma, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.alpha, self.beta, self.sigma, self.randvar)

#     def cdf(self,
#             plot=False,
#             threshold=1000,
#             interval = 1,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either cumulative distribution evaluation for some point or plot of Burr distribution.
#         """
#         _generator = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*(np.log10(x/o))**2)
#         if plot == True:
#             if interval<0:
#                 raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
#             x = np.linspace(0, interval, int(threshold))
#             y = np.array([_generator(self.alpha, self.sigma, self.beta, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.alpha, self.beta, self.sigma, self.randvar)

#     def pvalue(self, x_lower=0, x_upper=None):
#         """
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the Burr distribution evaluated at some random variable.
#         """
#         if x_upper == None:
#             x_upper = self.randvar
#         if x_lower>x_upper:
#             raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
#             _cdf_def  = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*pow((np.log10(x/o)),2))
#         return _cdf_def(self.alpha, self.beta, self.sigma, x_upper)-_cdf_def(self.alpha, self.beta, self.sigma, x_lower)

#     def mean(self):
#         """
#         Returns: Mean of the Burr distribution.
#         """
#         alpha = self.alpha; beta = self.beta; sigma = self.sigma
#         return sigma+(sigma/(sqrt(2*beta)))*ss.hermite(-1, (1+alpha)/(sqrt(2*beta)))

#     def median(self):
#         """
#         Returns: Median of the Burr distribution.
#         """
#         alpha = self.alpha; beta = self.beta; sigma = self.sigma
#         return sigma*np.exp((-alpha+sqrt(pow(alpha,2)+beta*np.log(16)))/(2*beta))

#     def mode(self):
#         """
#         Returns: Mode of the Burr distribution.
#         """
#         return self.beta/(self.alpha+1)

#     def var(self):
#         """
#         Returns: Variance of the Burr distribution.
#         """
#         mean = self.mean()
#         alpha = self.alpha; beta = self.beta; sigma = self.sigma
#         return (sigma**2+(2*sigma**2)/sqrt(2*beta)*ss.hermite(-1,(2+alpha)/sqrt(2*beta)))-mean**2

#     def skewness(self):
#         """
#         Returns: Skewness of the Burr distribution. 
#         """
#         if self.alpha>3:
#             return (4*sqrt(self.alpha-2))/(self.alpha-3)
#         return "undefined"

#     def kurtosis(self):
#         """
#         Returns: kurtosis of the Burr distribution
#         """
#         alpha = self.alpha
#         if alpha >4:
#             return 6*(5*alpha-11)/((alpha-3)*(alpha-4))
#         return "undefined"

#     def summary(self):
#         """
#         Returns: Summary statistic regarding the Burr distribution
#         """
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = " summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Dagum(Base):
    """
    This class contains methods concerning Dagum Distirbution. 
    Args:
    
        p_shape(float | x>0): shape parameter
        a_shape(float | x>0): shape parameter
        scale(float | x>0): scale parameter
        randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2019, March 14). Dagum distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 23:09, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Dagum_distribution&oldid=887692271
    """
    def __init__(self, p_shape, a_shape, scale, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable shoould be a positive number. Entered value:{}'.format(randvar))
         if p_shape<0 or a_shape<0 or scale<0:
            raise ValueError('shape and scale parameters should be a positive number. Entered value: p_shape={}, a_shape={}, scale={}'.format(p_shape, a_shape, scale))

        self.p_shape = p_shape
        self.a_shape = a_shape
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Dagum distribution.
        """
        _generator = lambda p,a,b,x: (a*p/x)*(np.power(x/b,a*p)/(np.power(pow((x/b),a)+1,p+1)
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))

            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.p_shape, self.a_shape, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)

        return _generator(self.p_shape, self.a_shape, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Dagum distribution.
        """
        _generator = lambda p,a,b,x: np.power((1+np.power(x/b,-a)),-p)
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.p_shape, self.a_shape,self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.p_shape, self.a_shape,self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Dagum distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda p,a,b,x: np.power((1+np.power(x/b,-a)),-p) if x>0 else 0
        return _cdf_def(self.p_shape, self.a_shape, self.scale, x_upper)-_cdf_def(self.p_shape, self.a_shape, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Dagum distribution.
        """
        a = self.a_shape
        p = self.p_shape
        b = self.scale
        if a>1:
            return (-b/a)*(ss.gamma(1/a)*ss.gamma(1/a+p))/ss.gamma(p)
        return "indeterminate"

    def median(self):
        """
        Returns: Median of the Dagum distribution.
        """
        a = self.a_shape
        p = self.p_shape
        b = self.scale
        return b*pow(-1+pow(2,1/p),-1/a)

    def mode(self):
        """
        Returns: Mode of the Dagum distribution.
        """
        a = self.a_shape
        p = self.p_shape
        b = self.scale
        return b*np.power((a*p-1)/(a+1), 1/a)

    def var(self):
        """
        Returns: Variance of the Dagum distribution.
        """
        a = self.a_shape
        p = self.p_shape
        b = self.scale
        if a>2:
            return (-pow(b,2)/pow(a,2))*(2*a*(ss.gamma(-2/a)*ss.gamma(2/a+p))/ss.gamma(p)+pow(ss.gamma(-1/a)*ss.gamma(1/a+p)/ss.gamma(p),2)) 
        return "indeterminate"

    def std(self):
        """
        Returns: Standard deviation of the Dagum distribution
        """
        if self.var() == "indeterminate":
            return "indeterminate"
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Dagum distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

        
class Davis(Base):
    """
    This class contains methods concerning Davis Distirbution. 
    Args:
    
        shape(float | x>0): shape parameter
        scale(float | x>0): scale parameter
        loc(float | x>0): location parameter
        randvar(float | x>loc): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2019, November 19). Davis distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 23:20, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Davis_distribution&oldid=927002685
    """
    def __init__(self, scale, shape, loc, randvar):
        if randvar>loc:
            raise ValueError('random variable should be greater than loc parameter')
         if scale<0 or shape<0 or loc<0:
            raise ValueError('shape, scale, and location parameters should be a positive number. Entered value: scale={}, shape={}, loc={}'.format(scale, shape, loc)

        self.scale = scale
        self.shape = shape
        self.loc = loc
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Davis distribution.
        """
        _generator = lambda b,n,mu, x: (pow(b,n)*pow(x-mu,-1-n))/((np.exp(b/(x-mu))-1)*ss.gamma(n)*ss.zeta(n))
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.scale, self.shape, self.loc, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.loc, self.randvar)

    # def cdf(self,
    #         plot=False,
    #         threshold=1000,
    #         interval = 1,
    #         xlim=None,
    #         ylim=None,
    #         xlabel=None,
    #         ylabel=None):
    #     """
    #     Args:
        
    #         interval(int): defaults to none. Only necessary for defining plot.
    #         threshold(int): defaults to 1000. Defines the sample points in plot.
    #         plot(bool): if true, returns plot.
    #         xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
    #         ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
    #         xlabel(string): sets label in x axis. Only relevant when plot is true. 
    #         ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
    #     Returns: 
    #         either cumulative distribution evaluation for some point or plot of Burr distribution.
    #     """
    #     _generator = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*(np.log10(x/o))**2)
    #     if plot == True:
    #         if interval<0:
    #             raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
    #         x = np.linspace(0, interval, int(threshold))
    #         y = np.array([_generator(self.alpha, self.sigma, self.beta, i) for i in x])
    #         return super().plot(x, y, xlim, ylim, xlabel, ylabel)
    #     return _generator(self.alpha, self.beta, self.sigma, self.randvar)

    # def pvalue(self, x_lower=0, x_upper=None):
    #     """
    #     Args:

    #         x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
    #         x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

    #         Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
    #         Otherwise, the default random variable is x.

    #     Returns:
    #         p-value of the Burr distribution evaluated at some random variable.
    #     """
    #     if x_upper == None:
    #         x_upper = self.randvar
    #     if x_lower>x_upper:
    #         raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
    #     _cdf_def  = lambda a,b,o,x: 1- np.exp(-a*np.log10(x/a)-b*(np.log10(x/o))**2)
    #     return _cdf_def(self.alpha, self.beta, self.sigma, x_upper)-_cdf_def(self.alpha, self.beta, self.sigma, x_lower)

    def mean(self):
        """
        Returns: Mean of the Davis distribution.
        """
        b,n,mu = self.scale, self.shape, self.loc
        if n>2:
            return mu+(b*ss.zeta(n-1))/((n-1)*ss.zeta(n))
        return "indeterminate"

    def var(self):
        """
        Returns: Variance of the Davis distribution.
        """
        b,n,mu = self.scale, self.shape, self.loc
        if n>3:
            return (pow(b,2)*(-(n-2)*pow(ss.zeta(n-1),2)+(n-1)*ss.zeta(n-2)*ss.zeta(n)))/((n-2)*pow(n-1,2)*pow(ss.zeta(n),2))
        return "indeterminate"

    def std(self):
        """
        Returns: Standard deviation of the Davis distribution
        """
        if self.var() == "indeterminate":
            return "indeterminate"
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Davis distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Rayleigh(Base):
    """
    This class contains methods concerning Rayleigh Distirbution. 
    Args:
    
        scale(float | x>0): scale
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 30). Rayleigh distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:37, January 8, 2021, from https://en.wikipedia.org/w/index.php?title=Rayleigh_distribution&oldid=997166230

    - Weisstein, Eric W. "Rayleigh Distribution." From MathWorld--A Wolfram Web Resource. 
    https://mathworld.wolfram.com/RayleighDistribution.html
    """
    def __init__(self, scale, randvar):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value: {}'.format(randvar))
        if scale<0:
            raise ValueError('scale parameter should be a positive number.')

        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Rayleigh distribution.
        """
        _generator = lambda sig,x: (x/pow(sig,2))*np.exp(pow(-x,2)/(2*pow(sig,2)))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Rayleigh distribution.
        """
        _generator = lambda sig,x: 1-np.exp(-x**2/(2*sig**2))
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Rayleigh distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda sig,x: 1-np.exp(-x**2/(2*sig**2))
        return _cdf_def(self.scale, x_upper)-_cdf_def(self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Rayleigh distribution.
        """
        return self.scale*sqrt(np.pi/2)

    def median(self):
        """
        Returns: Median of the Rayleigh distribution.
        """
        return self.scale*sqrt(2*np.log(2))

    def mode(self):
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return self.scale

    def var(self):
        """
        Returns: Variance of the Rayleigh distribution.
        """
        return  (4-np.pi)/2*pow(self.scale, 2)

    def std(self):
        """
        Returns: Standard deviation of the Rayleigh distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Rayleigh distribution. 
        """
        return (2*sqrt(np.pi)*(np.pi-3))/pow((4-np.pi), 3/2)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Rayleigh distribution. 
        """
        return -(6*pow(np.pi,2)-24*np.pi+16)/pow(4-np.pi,*2)

    def entropy(self):
        """
        Returns: differential entropy of the Rayleigh distribution.

        Reference: Park, S.Y. & Bera, A.K.(2009). Maximum entropy autoregressive conditional heteroskedasticity model. Elsivier. 
        link: http://wise.xmu.edu.cn/uploadfiles/paper-masterdownload/2009519932327055475115776.pdf
        """
        return 1+np.log(self.scale/sqrt(2))+(np.euler_gamma/2)

    def summary(self):
        """
        Returns: Summary statistic regarding the Rayleigh distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


# class Hypoexponential(Base):
#     """
#     This class contains methods concerning Hypoexponential Distirbution. 
#     Args:
    
#         rates()
#         randvar(float | x>0): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - entropy for differential entropy of the distribution.
#         - summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2020, December 13). Hypoexponential distribution. In Wikipedia, The Free Encyclopedia. 
#     Retrieved 12:21, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Hypoexponential_distribution&oldid=994035019
#     """
#     def __init__(self, randvar=0.5,*args):
#         if randvar<0:
#             raise ValueError('random variable shoould be a positive number. Entered value:{}'.format(randvar))
#         self.args = [i for i in args]
#         self.randvar = randvar

#     def pdf(self):
#         return "no other simple form. Currently unsupported"

#     def cdf(self):
#         return "no other simple form. Currently unsupported"

#     def mean(self):
#         """
#         Returns: Mean of the Hypoexponential distribution.
#         """
#         return sum([1/x for x in self.args])

#     def median(self):
#         """
#         Returns: Median of the Hypoexponential distribution.
#         """
#         return "Gneral closed-form does not exist"

#     def mode(self):
#         """
#         Returns: Mode of the Hypoexponential distribution.
#         """
#         a = self.a_shape
#         p = self.p_shape
#         b = self.scale
#         return b*pow((a*p-1)/(a+1), 1/a)

#     def var(self):
#         """
#         Returns: Variance of the Hypoexponential distribution.
#         """
#         return "currently unsupported"

#     def skewness(self):
#         """
#         Returns: Skewness of the Hypoexponential distribution. 
#         """
#         return 2*sum([1/pow(x,3) for x in self.args])/pow(sum([1/pow(x,2) for x in self.args]), 3/2)

#     def kurtosis(self):
#         """
#         Returns: kurtosis of the Hypoexponential distribution
#         """
#         return "no simple closed form"

#     def summary(self):
#         """
#         Returns: Summary statistic regarding the Hypoexponential distribution
#         """
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = " summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Benktander_T1(Base):
    """
    This class contains methods concerning Benktander Type1 Distirbution. 
    Args:
    
        a(float | x>0): shape parameter
        b(float | x>0): shape parameter
        randvar(float | x>=1): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2018, August 2). Benktander type I distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:36, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Benktander_type_I_distribution&oldid=853042641
    """
    def __init__(self, a, b, scale, randvar=1.5):
        if randvar<1:
            raise ValueError('random variable shoould be a positive number, not less than 1. Entered value:{}'.format(randvar))
         if a<0 or b<0:
            raise ValueError('parameters should be a positive number. Entered value: a={}, b={}'.format(a,b))

        self.a = a
        self.b = b
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Benktander Type1 distribution.
        """
        _generator = lambda a,b,x: ((1+(2*b*np.log10(x)/a))*(1+a+2*np.log10(x)-2*b/a)*np.power(x, -(2+a+b*np.log10(x))) # log base 10, validate this
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Benktander Type1 distribution.
        """
        _generator = lambda a,b,x: 1-(1+(2*b/a)*np.log10(x))*np.power(x,-(a+1+b*np.log10(x)))
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Benktander Type1 distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: 1-(1+(2*b/a)*np.log10(x))*np.power(x,-(a+1+b*np.log10(x))) if x>0 else 0
        return _cdf_def(self.a, self.b, x_upper)-_cdf_def(self.a, self.b, x_lower)

    def mean(self):
        """
        Returns: Mean of the Benktander Type1 distribution.
        """
        return 1+1/self.a

    def var(self):
        """
        Returns: Variance of the Benktander Type1 distribution.
        """
        a = self.a
        b = self.b
        x = self.x
        return (-sqrt(b)+a*np.exp(pow(a-1,2)/(4*b))*sqrt(np.pi)*ss.erfc((a-1)/(2*sqrt(b))))/(pow(a,2)*sqrt(b))

    def std(self):
        """
        Returns: Standard deviation of the Benktander Type1 distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Benktander Type1 distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Benktander_T2(Base):
    """
    This class contains methods concerning Benktander Type 2 Distirbution. 
    Args:
    
        a(float | x>0): shape parameter
        b(float | x>0): shape parameter
        randvar(float | x>=1): random variable. Optional. Use when cdf and pdf or p value of interest is desired.

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, June 11). Benktander type II distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 12:43, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Benktander_type_II_distribution&oldid=962001463
    """
    def __init__(self, a, b, scale, randvar=1.5):
        if randvar<1:
            raise ValueError('random variable shoould be a positive number, not less than 1. Entered value:{}'.format(randvar))
        if a<0 or b<0 or b>1:
            raise ValueError('parameters a amd b should be a positive number where b is not greater than 1. Entered value: a={}, b={}'.format(a,b))

        self.a = a
        self.b = b
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Benktander type 2 distribution.
        """
        _generator = lambda a,b,x: np.exp(a/b*(1-x**b))*np.power(x,b-2)*(a*x**b-b+1)
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Benktander type 2 distribution.
        """
        _generator = lambda a,b,x: 1- np.power(x, b-1)*np.exp(a/b*(1-x**b))
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.b, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.b, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Benktander type 2 distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: 1- np.power(x, b-1)*np.exp(a/b*(1-x**b)) if x>0 else 0
        return _cdf_def(self.a, self.b, x_upper)-_cdf_def(self.a, self.b, x_lower)

    def mean(self):
        """
        Returns: Mean of the Benktander type 2 distribution.
        """
        return 1+1/self.a

    def median(self):
        """
        Returns: Median of the Benktander type 2 distribution.
        """
        a = self.a
        b = self.b
        if b ==1:
            return np.log10(2)/a + 1
        return pow(((1-b)/a)*ss.lambertw((pow(2, b/(1-b))*a*np.exp(1/(1-b))/(1-b))), 1/b)
    

    def mode(self):
        """
        Returns: Mode of the Benktander type 2 distribution.
        """
        return 1

    def var(self):
        """
        Returns: Variance of the Benktander type 2 distribution.
        """
        a = self.a
        p = self.b
        return (-b+2*a*np.exp(a/b)*ss.expn(1-1/b. a/b))/(pow(a,2)*b)

    def std(self):
        """
        Returns: Standard deviation of the Benktander Type 2 distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Benktander type 2 distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

class Cauchy_log(Base):
    """
    This class contains methods concerning log-Cauchy Distirbution. 
    Args:
    
        mu(float | x>0): shape parameter
        scale(float | x>0): scale parameter
        randvar(float | x>loc): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, December 7). Log-Cauchy distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:15, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Log-Cauchy_distribution&oldid=992914094
    """
    def __init__(self, mu, scale, randvar):
        if randvar<0:
            raise ValueError('random variable should be a positive number')
        if mu<0 or scale<0:
            raise ValueError('mu and scale parameters should be a positive number. Entered value: mu={}, scale={}'.format(mu, scale)

        self.mu = mu
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of log-Cauchy distribution.
        """
        _generator = lambda mu, sig, x: (1/(x*np.pi))*(sig/((np.log(x)-mu)**2+sig**2))
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.mu, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.mu, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of log-Cauchy distribution.
        """
        _generator = lambda mu, sig, x: (1/np.pi)*np.arctan((np.log(x)-mu)/sig)+0.5 if x>0 else 0
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.mu, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.mu, self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the log-Cauchy distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda mu, sig, x: (1/np.pi)*np.arctan((np.log(x)-mu)/sig)+0.5 if x>0 else 0
        return _cdf_def(self.mu, self.scale, x_upper)-_cdf_def(self.mu, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the log-Cauchy distribution.
        """
        return "infinite"

    def median(self):
        """
        Returns: Median of the log-Cauchy distribution.
        """
        return np.exp(self.mu)

    def var(self):
        """
        Returns: Variance of the log-Cauchy distribution.
        """
        return "infinite"  

    def std(self):
        """
        Returns: Standard deviation of the log-Cauchy distribution
        """
        return "infinite"

    def skewness(self):
        """
        Returns: Skewness of the log-Cauchy distribution. 
        """
        return "does not exist"

    def kurtosis(self):
        """
        Returns: kurtosis of the log-Cauchy distribution
        """
        return "does not exist"

    def summary(self):
        """
        Returns: Summary statistic regarding the log-Cauchy distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Laplace_log(Base):
    """
    This class contains methods concerning log-Laplace Distirbution. 
    Args:
    
        loc(float | x>0): location parameter
        scale(float | x>0): scale parameter
        randvar(float | x>loc): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, October 19). Log-Laplace distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:24, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Log-Laplace_distribution&oldid=984391227
    """
    def __init__(self, loc, scale, randvar):
         if scale<0:
            raise ValueError('scale parameters should be a positive number. Entered value: scale={}'.format(scale)

        self.loc = loc
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of log-Laplace distribution.
        """
        _generator = lambda mu, b, x: 1/(2*b*x)*np.exp(-abs(np.log(x)-mu)/b)
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of log-Laplace distribution.
        """
        _generator = lambda mu, b, x:0.5*(1+np.sign(np.log(x)-mu)*(1-np.exp(-abs(np.log(x)-mu)/b))) if x>0 else 0
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the log-Laplace distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda mu, b, x:0.5*(1+np.sign(np.log(x)-mu)*(1-np.exp(-abs(np.log(x)-mu)/b))) if x>0 else 0
        return _cdf_def(self.loc, self.scale, x_upper)-_cdf_def(self.loc, self.scale, x_lower)


class Logistic_log(Base):
    """
    This class contains methods concerning Log logistic Distirbution. 
    Args:
    
        shape(float | x>0): shape parameter
        scale(float | x>0): scale parameter
        randvar(float | x>loc): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2019, November 19). Davis distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 23:20, January 15, 2021, from https://en.wikipedia.org/w/index.php?title=Davis_distribution&oldid=927002685
    """
    def __init__(self, scale, shape, randvar=0.5):
        if randvar<0:
            raise ValueError('random variable should be a positive number')
        if scale<0 or shape<0:
            raise ValueError('shape, scale, and location parameters should be a positive number. Entered value: scale={}, shape={}'.format(scale, shape)

        self.scale = scale
        self.shape = shape
        self.randvar = randvar

    def pdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Log logistic distribution.
        """
        _generator = lambda a,b,x: (b/a)*np.power(x/a, b-1)/(1+(x/a)**b)**2
        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, 1, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval = 1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Log logistic distribution.
        """
        _generator = lambda a,b,x: 1/(1+np.power(x/a, -b))
        if plot == True:
            if interval<0:
                raise ValueError('interval parameter should be a positive number. Entered Value {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Log logistic distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: 1/(1+np.power(x/a, -b))
        return _cdf_def(self.scale, self.shape, x_upper)-_cdf_def(self.scale, self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Log logistic distribution.
        """
        if self.shape>1:
            return (self.scale*np.pi/self.shape)/np.sin(np.pi/self.shape)
        return "undefined"

    def median(self):
        """
        Returns: Median of the log Logistic distribution.
        """
        return self.scale

    def mode(self):
        """
        Returns: Mode of the log Logistic distribution.
        """
        a = self.scale
        b = self.shape
        if b>1:
            return a*np.power((b-1)/(b+1), 1/b)
        return 0

    def var(self):
        """
        Returns: Variance of the Log logistic distribution.
        """
        a = self.scale
        b = self.shape
        if b>2:
            return pow(a,2)*(2*b/np.sin(2*b)-pow(b,2)/pow(np.sin(b),2))
        return "undefined"

    def std(self):
        """
        Returns: Standard deviation of the Log logistic distribution
        """
        if self.var() == "undefined":
            return "undefined"
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Log logistic distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Chisq_inv(Base):
    """
    This class contains methods concerning Inverse Chi-squared Distirbution. 
    Args:
    
        df(int | x>0): degrees of freedom
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, October 6). Inverse-chi-squared distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 16:21, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=Inverse-chi-squared_distribution&oldid=982193912

    - Mathematica (2021): InverseChiSquareDistribution. Retrieved from: https://reference.wolfram.com/language/ref/InverseChiSquareDistribution.html
    """
    def __init__(self, df, randvar):
        if randvar<0:
            raise ValueError('random variable should be a positive number. Entered value: {}'.format(randvar))
        if df<0:
            raise ValueError('df parameter should be a positive number.')

        self.df = df
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Inverse Chi-squared distribution.
        """
        _generator = lambda df, x: np.power(2,-df/2)/ss.gamma(df/2)*np.power(x,-df/2-1)*np.exp(-1/(2*x))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.df, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.df, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None): 
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Rayleigh distribution.
        """
        _generator = lambda df, x: ss.gammainc(df/2,1/(2*x))/ss.gamma(df/2)
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.df, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.df, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Inverse Chi-squared distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda df, x: ss.gammainc(df/2,1/(2*x))/ss.gamma(df/2)
        return _cdf_def(self.df, x_upper)-_cdf_def(self.df, x_lower)

    def mean(self):
        """
        Returns: Mean of the Inverse Chi-squared distribution.
        """
        if self.df>2:
            return 1/(df-2)
        return "undefined"

    def median(self):
        """
        Returns: Approximation of the Median of the Inverse Chi-squared distribution.
        """
        return 1/(self.df*(1-2/(9*self.df))**3)

    def mode(self):
        """
        Returns: Mode of the Rayleigh distribution.
        """
        return 1/(self.df+2)

    def var(self):
        """
        Returns: Variance of the Inverse Chi-squared distribution.
        """
        df = self.df
        if df>4:
            return 2/((df-2)**2*(df-4))
        return "undefined"

    def std(self):
        """
        Returns: Standard deviation of the Log logistic distribution
        """
        if self.var() == "undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Inverse Chi-squared distribution. 
        """
        df = self.df
        if df>6:
            return 1/(df-6)*sqrt(2*(df-4))
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Inverse Chi-squared distribution. 
        """
        df = self.df
        if df>8:
            (12*(5*df-22))/((df-6)*(df-8))

    def entropy(self):
        """
        Returns: differential entropy of the Inverse Chi-squared distribution.
        """
        df = self.df
        return df/2+np.log((df/2)*ss.gamma(df/2))-(1+df/2)*ss.digamma(df/2)

    def summary(self):
        """
        Returns: Summary statistic regarding the Inverse Chi-squared distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Levy(Base):
    """
    This class contains methods concerning Levy Distirbution. 
    Args:
    
        scale(float | x>0): scale parameter
        loc (float | x>0): location parameter
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 9). Lévy distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 16:35, January 16, 2021, from https://en.wikipedia.org/w/index.php?title=L%C3%A9vy_distribution&oldid=999292486
    """
    def __init__(self, scale, loc, randvar):
        if randvar<scale:
            raise ValueError('random variable should be not be less than the location parameter. Entered value: {}'.format(randvar))
        if scale<0:
            raise ValueError('scale parameter should be a positive number. Enetered value for scale =  {}'.format(scale))

        self.scale = scale
        self.loc = loc
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Levy distribution.
        """
        _generator = lambda mu, c ,x: sqrt(c/(2*np.pi))*np.exp(-c/(2*(x-mu)))/np.power(x-mu, 3/2)

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Levy distribution.
        """
        _generator = lambda mu, c, x: ss.erfc(sqrt(c/(2*(x-mu))))
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda mu, c, x: ss.erfc(sqrt(c/(2*(x-mu))))
        return _cdf_def(self.loc, self.scale, x_upper)-_cdf_def(self.loc, self.scale, x_lower)

    def mean(self):
        """
        Returns: Mean of the Levy distribution.
        """
        return "infinity"

    def median(self):
        """
        Returns: Median of the Levy distribution.
        """
        return self.loc+self.scale/2*ss.erfcinv(0.5)**2

    def mode(self):
        """
        Returns: Mode of the Levy distribution.
        """
        return self.loc+self.scale/3

    def var(self):
        """
        Returns: Variance of the Levy distribution.
        """
        return "infinity"

    def std(self):
        """
        Returns: Standard deviation of the Levy distribution
        """
        return "infinity"

    def skewness(self):
        """
        Returns: Skewness of the Levy distribution. 
        """
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Levy distribution. 
        """
        return "undefined"

    def entropy(self):
        """
        Returns: differential entropy of the Levy distribution.
        """
        return (1+3*np.euler_gamma+np.log(16*np.pi*self.scale**2))/2

    def summary(self):
        """
        Returns: Summary statistic regarding the Levy distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Nakagami(Base):
    """
    This class contains methods concerning Nakagami Distirbution. 
    Args:
    
        shape(float | x>0): shape parameter
        spread(float | x>0): spread parameter
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 11). Nakagami distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:29, January 17, 2021, from https://en.wikipedia.org/w/index.php?title=Nakagami_distribution&oldid=999782097
    """
    def __init__(self, shape, spread, randvar):
        if randvar<=0:
            raise ValueError('random variable should be a positive number. Entered value: {}'.format(randvar))
        if shape<0.5:
            raise ValueError('shape parameter should not be less then 0.5. Enetered value for shape =  {}'.format(shape))
        if spread<=0:
            raise ValueError('spread should be a positive number. Entered value: {}'.format(spread))

        self.shape = shape
        self.spread = spread
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Nakagami distribution.
        """
        _generator = lambda m, omega, x: (2*pow(m,m))/(ss.gamma(m)*pow(omega,m))*pow(x, 2*m-1)*np.exp(-m/omega*pow(x,2))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.shape, self.spread, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.spread, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Nakagami distribution.
        """
        _generator = lambda m, omega, x: ss.gammainc(m, (m/omega)*pow(x,2))/ss.gamma(m)
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.shape, self.spread, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.shape, self.spread, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda m, omega, x: ss.gammainc(m, (m/omega)*pow(x,2))/ss.gamma(m)
        return _cdf_def(self.shape, self.spread, x_upper)-_cdf_def(self.shape, self.spread, x_lower)

    def mean(self):
        """
        Returns: Mean of the Nakagami distribution.
        """
        m, omega = self.shape, self.spread
        return (ss.gamma(m+0.5)/ss.gamma(m))*sqrt(omega/m)

    def median(self):
        """
        Returns: Median of the Nakagami distribution.
        """
        return "no simple closed form"

    def mode(self):
        """
        Returns: Mode of the Nakagami distribution.
        """
        m, omega = self.shape, self.spread
        return sqrt(2)/2*sqrt((2*m-1)*omega/m)

    def var(self):
        """
        Returns: Variance of the Nakagami distribution.
        """
        m, omega = self.shape, self.spread
        return omega*(1-(1/m)*pow(ss.gamma(m+0.5)/ss.gamma(m),2))

    def std(self):
        """
        Returns: Standard deviation of the Nakagami distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Nakagami distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Lomax(Base):
    """
    This class contains methods concerning Lomax Distirbution. 
    Args:
    
        shape(float | x>0): shape parameter
        scale(float | x>0): scale parameter
        randvar(float | x>=0): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 11). Lomax distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 09:29, January 17, 2021, from https://en.wikipedia.org/w/index.php?title=Nakagami_distribution&oldid=999782097
    """
    def __init__(self, shape, scale, randvar):
        if randvar<=0:
            raise ValueError('random variable should be a positive number. Entered value: {}'.format(randvar))
        if scale<=0 or shape<=0:
            raise ValueError('shape and scale parameters should be a positive number.')

        self.shape = shape
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Lomax distribution.
        """
        _generator = lambda lambda_, alpha, x: alpha/lambda_*pow(1+x/lambda_, -(alpha+1))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Lomax distribution.
        """
        _generator = lambda lambda_, alpha, x: 1 - pow(1+x/lambda_, -alpha)
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.scale, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda lambda_, alpha, x: 1 - pow(1+x/lambda_, -alpha) if x>=0 else 0
        return _cdf_def(self.scale, self.shape, x_upper)-_cdf_def(self.scale, self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Lomax distribution.
        """
        if self.shape>1:
            return self.scale/(self.shape+1)
        return "undefined"

    def median(self):
        """
        Returns: Median of the Lomax distribution.
        """
        return self.scale*(pow(2,1/self.shape)-1)

    def mode(self):
        """
        Returns: Mode of the Lomax distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Lomax distribution.
        """
        alpha, lambda_ = self.shape, self.scale
        if alpha>2:
            return (pow(lambda_, 2)*alpha)/(pow(alpha-1,2)*(alpha-2))
        elif alpha>1 and alpha <=2:
            return np.inf
        else:
            return "undefined"

    def std(self):
        """
        Returns: Standard deviation of the Lomax distribution
        """
        if self.var() == "undefined":
            return "undefined"
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Lomax distribution. 
        """
        alpha, lambda_ = self.shape, self.scale
        if alpha >3:
            return (2*(1+alpha))/(alpha-3)*sqrt((alpha-2)/alpha)
        return "undefined"

    def kurtosis(self):
        """
        Returns: Kurtosis of the Lomax distribution. 
        """
        alpha, lambda_ = self.shape, self.scale
        if alpha>4:
            return (6*(pow(alpha, 3)+pow(alpha,2)-6*alpha-2))/(alpha*(alpha-3)*(alpha-4))
        return "undefined"

    def summary(self):
        """
        Returns: Summary statistic regarding the Lomax distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Gumbel_T1(Base):
    """
    This class contains methods concerning Gumbel Distirbution. 
    Args:
    
        loc(float): loc parameter
        scale(float | x>0): scale parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 26). Gumbel distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:02, January 17, 2021, from https://en.wikipedia.org/w/index.php?title=Gumbel_distribution&oldid=990718796
    """
    def __init__(self, loc, scale, randvar):
        if scale<=0:
            raise ValueError('scale parameters should be a positive number.')

        self.loc = loc
        self.scale = scale
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Gumbel distribution.
        """
        def _generator(mu, beta, x):
            z = (x-mu)/beta
            return (1/beta)*np.exp(-z*np.exp(-z))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.mu, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.mu, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Gumbel distribution.
        """
        _generator = lambda mu, beta, x: np.exp(-np.exp(-(x-mu)/beta))
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.loc, self.mu, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.mu, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Erlang distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda mu, beta, x: np.exp(-np.exp(-(x-mu)/beta))
        return _cdf_def(self.loc, self.mu, x_upper)-_cdf_def(self.loc, self.mu, x_lower)

    def mean(self):
        """
        Returns: Mean of the Gumbel distribution.
        """
        return self.loc+self.scale*np.euler_gamma
    
    def median(self):
        """
        Returns: Median of the Gumbel distribution.
        """
        return self.loc - self.scale*np.log(np.log(2))

    def mode(self):
        """
        Returns: Mode of the Gumbel distribution.
        """
        return self.loc

    def var(self):
        """
        Returns: Variance of the Gumbel distribution.
        """
        return (pow(np.pi,2)/6)*pow(self.scale,2)

    def std(self):
        """
        Returns: Standard deviation of the Gumbel distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Approximation of the Skewness of the Gumbel distribution. 
        """
        return 1.14

    def kurtosis(self):
        """
        Returns: Kurtosis of the Gumbel distribution. 
        """
        return 12/5
    
    def entropy(self):
        """
        Returns: differential entropy of the Gumbel distribution.
        """
        return np.log(self.scale)+np.euler_gamma+1

    def summary(self):
        """
        Returns: Summary statistic regarding the Gumbel distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Gumbel_T2(Base):
    """
    This class contains methods concerning Gumbel Type 2 Distirbution. 
    Args:
    
        a(float): parameter
        shape(float): scale parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2018, April 13). Type-2 Gumbel distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:11, January 17, 2021, from https://en.wikipedia.org/w/index.php?title=Type-2_Gumbel_distribution&oldid=836161575
    """
    def __init__(self, a, shape, randvar):
        self.a = a
        self.shape = shape
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Gumbel Type 2 distribution.
        """
        _generator = lambda a,b,x: pow(a*b*x, -a-1)*np.exp(-b*pow(x,-a))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Gumbel Type 2 distribution.
        """
        _generator = lambda a,b,x: np.exp(-b*pow(x,-a))
        if plot == True:
            x = np.linspace(0, interval, int(threshold))
            y = np.array([_generator(self.a, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Gumbel type 2 distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def  = lambda a,b,x: np.exp(-b*pow(x,-a))
        return _cdf_def(self.a, self.shape, x_upper)-_cdf_def(self.a, self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Gumbel Type 2 distribution.
        """
        return pow(self.shape, 1/self.a)*ss.gamma(1-1/self.a)

    def var(self):
        """
        Returns: Variance of the Gumbel Type 2 distribution.
        """
        return pow(self.shape, 2/self.a)*(ss.gamma(1-1/self.a)-pow(ss.gamma(1-1/self.a),2))

    def std(self):
        """
        Returns: Standard deviation of the Gumbel Type 2 distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Gumbel Type 2 distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# supported on the whole real line 
class Fisher_z(Base):
    """
    This class contains methods concerning Fisher's z-Distirbution. 
    Args:
    
        df1(float): degrees of freedom
        df2(float): degrees of freedom
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Note: Fisher's z-distribution is the statistical distribution of half the log of an F-distribution variate:
    z = 1/2*log(F)

    Reference:
    - Wikipedia contributors. (2020, December 15). Fisher's z-distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 02:33, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Fisher%27s_z-distribution&oldid=994427156
    """
    def __init__(self, df1, df2, randvar=0):
        self.df1 = df1
        self.df2 = df2
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Fisher's z-distribution.
        """
        _generator = lambda df1, df2, x: pow(2*df1, df1/2)*pow(df2, df2/2)*np.exp(df1*x)/(ss.beta(df1/2,df2/2)*pow(df1*np.exp(2*x)+df2,(df1+df2)/2))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.a, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.a, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Fisher's z-distribution.
        """
        def _generator(df1, df2, x,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None): 

            f = F(df1, df2, np.exp(2*x))
            if plot == True:
                return f.cdf(plot, threshold, interval, xlim, ylim, xlabel, ylabel)
            return f.cdf()

        if plot == True:
            _generator(self.df, self.df2, self.randvar)
        return _generator(self.df, self.df2, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Fisher's z-distribution evaluated at some random variable.
        """
        def _cdf_def(df1,df2,x):
            f = F(df1, df2, x)
            return f.pvalue(x_lower, x_upper)

        return _cdf_def(self.df1, self.df2, self.randvar)

    def summary(self):
        """
        Returns: Summary statistic regarding the Fisher's z-distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Laplace_asym(Base):
    """
    This class contains methods concerning Asymmetric Laplace Distirbution. 
    Args:
    
        loc(float): location parameter
        scale(float): scale parameter
        asym(float): asymmetry parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2018, April 13). Type-2 Gumbel distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 14:11, January 17, 2021, from https://en.wikipedia.org/w/index.php?title=Type-2_Gumbel_distribution&oldid=836161575
    """
    def __init__(self, loc, scale, asym, randvar=0):
        if scale<=0 or asym <=0:
            raise ValueError('scale and asym parameters should be a positive number. Entered values: scale = {}, asym = {}'.format(scale, asym))
        self.loc = loc
        self.scale = scale
        self.asym = asym
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Asymmetric Laplace distribution.
        """
        _generator = lambda m,l,k,x: 1/(k+1/k)*np.exp(-(x-m)*l*np.sign(x-m)*pow(k,np.sign(x-m)))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, self.asym, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.asym, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Asymmetric Laplace distribution.
        """

        def _generator(loc, scale, asym, x):
            if x<=loc:
                return pow(asym,2)/(1+pow(asym,2))*np.exp((scale/asym)*(x-loc))
            return 1-(1/(1+pow(asym,2)))*np.exp(-scale*asym*(x-loc))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, self.asym,, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.asym, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Asymmetric Laplace distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        def _cdf_def(loc, scale, asym, x):
            if x<=loc:
                return pow(asym,2)/(1+pow(asym,2))*np.exp((scale/asym)*(x-loc))
            return 1-(1/(1+pow(asym,2)))*np.exp(-scale*asym*(x-loc))

        return _cdf_def(self.loc, self.scale, self.asym, x_upper)-_cdf_def(self.loc, self.scale, self.asym, x_lower)

    def mean(self):
        """
        Returns: Mean of the Asymmetric Laplace distribution.
        """
        m = self.loc
        k = self.asym
        return m+(1-pow(k,2))/(self.scale*k)

    def median(self):
        """
        Returns: Median of the Asymmetric Laplace distribution.
        """
        m,k,l = self.loc, self.asym, self.scale
        if k>1:
            return m+(k/l)*log(10, (1+pow(k,2))/(2*pow(k,2)))
        if k<1:
            return m+(1/(l*k))*log(10, (1+pow(k,2))/2)
        return "undefined"

    def var(self):
        """
        Returns: Variance of the Asymmetric Laplace distribution.
        """
        k,l = self.asym, self.scale
        return (1+pow(k,4))/(pow(l,2)*pow(k,2))

    def std(self):
        """
        Returns: Standard deviation of the Asymmetric Laplace distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Approximation of the Skewness of the Asymmetric Laplace distribution. 
        """
        k = self.asym
        return (2*(1+pow(k,6)))/pow(pow(k,4)+1, 3/2)

    def kurtosis(self):
        """
        Returns: Kurtosis of the Asymmetric Laplace distribution. 
        """
        k = self.asym
        return (6*(1+pow(k,8)))/pow(1+pow(k,4),2)
    
    def entropy(self):
        """
        Returns: differential entropy of the Asymmetric Laplace distribution.
        """
        k = self.asym
        l = self.scale
        return np.log(np.e*(1+pow(k,2)/(k*l)))


    def summary(self):
        """
        Returns: Summary statistic regarding the Asymmetric Laplace distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# add cdf, pvalue, and MGF 
class GH(Base):
    """
    This class contains methods concerning Generalized Hyperbolic Distirbution. 
    Args:
    
        lambda_(float): lambda_ parameter
        alpha(float): alpha parameter
        asym(float): asymmetry parameter
        scale (float): scale parameter
        loc(float): location 

        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Note:
    
        - Related Distributions:

            - X~GH(-df/2,0,0,sqrt(df), mu) - T distribution with df as degrees of freedom and mu as mean.
            - X~GH(1,α,β,δ,μ) - Hyperbolic distribution
            - X~GH(-1/2,α,β,δ,μ) - normal-inverse Gaussian distribution
            - X~GH(?,?,?,?,?) - normal-inverse chi-squared distirbution
            - X~GH(?,?,?,?,?) - normal-inverse gamma distirbution
            - X~GH(λ,α,β,δ,0,γ) - variance-gamma distribution
            - X~GH(1,1,0,0,μ) - Laplace distribution with location parameter μ and scale parameter 1.

    Reference:
    - Wikipedia contributors. (2020, December 2). Generalised hyperbolic distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:12, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Generalised_hyperbolic_distribution&oldid=991884528
    - Wolfram (n.d.) https://demonstrations.wolfram.com/GeneralizedHyperbolicDistribution/
    - 
    """
    def __init__(self, lambda_, alpha, loc, scale, asym, randvar=0):
        self._lamba = _lamba
        self.alpha = alpha
        self.loc = loc
        self.scale = scale
        self.asym = asym
        self.randvar = randvar
        self.gamma = sqrt(pow(alpha,2)-pow(asym,2))

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Generalized Hyperbolic distribution.
        """
        _generator = lambda λ,α,β,δ,μ,γ,x: pow(γ/δ, λ)/(sqrt(2*np.pi)*ss.kn(λ, δ*γ))*np.exp(β*(x-μ))*(ss.kn(λ-0.5, α*sqrt(δ**2+pow(x-μ, 2))))/pow(sqrt(δ**2+pow(x-μ, 2))/α,0.5-λ)

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.lambda_, self.alpha, self.beta, self.scale, self.loc, self.gamma, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.lambda_, self.alpha, self.beta, self.scale, self.loc, self.gamma, self.randvar)

    # def cdf(self,
    #         plot=False,
    #         threshold=1000,
    #         interval=1,
    #         xlim=None,
    #         ylim=None,
    #         xlabel=None,
    #         ylabel=None):
    #     """
    #     Args:
        
    #         interval(int): defaults to none. Only necessary for defining plot.
    #         threshold(int): defaults to 1000. Defines the sample points in plot.
    #         plot(bool): if true, returns plot.
    #         xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
    #         ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
    #         xlabel(string): sets label in x axis. Only relevant when plot is true. 
    #         ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
    #     Returns: 
    #         either cumulative distribution evaluation for some point or plot of Generalized Hyperbolic distribution.
    #     """

    #     def _generator(loc, scale, asym, x):
    #         if x<=loc:
    #             return pow(asym,2)/(1+pow(asym,2))*np.exp((scale/asym)*(x-loc))
    #         return 1-(1/(1+pow(asym,2)))*np.exp(-scale*asym*(x-loc))

    #     if plot == True:
    #         x = np.linspace(-interval, interval, int(threshold))
    #         y = np.array([_generator(self.loc, self.scale, self.asym,, i) for i in x])
    #         return super().plot(x, y, xlim, ylim, xlabel, ylabel)
    #     return _generator(self.loc, self.scale, self.asym, self.randvar)

    # def pvalue(self, x_lower=0, x_upper=None):
    #     """
    #     Args:

    #         x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
    #         x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

    #         Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
    #         Otherwise, the default random variable is x.

    #     Returns:
    #         p-value of the Generalized Hyperbolic distribution evaluated at some random variable.
    #     """
    #     if x_upper == None:
    #         x_upper = self.randvar
    #     if x_lower>x_upper:
    #         raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
    #     def _cdf_def(loc, scale, asym, x):
    #       if x<=loc:
    #           return pow(asym,2)/(1+pow(asym,2))*np.exp((scale/asym)*(x-loc))
    #       return 1-(1/(1+pow(asym,2)))*np.exp(-scale*asym*(x-loc)

    #     return _cdf_def(self.loc, self.scale, self.asym, x_upper)-_cdf_def(self.loc, self.scale, self.asym, x_lower)

    def mean(self):
        """
        Returns: Mean of the Generalized Hyperbolic distribution.
        """
        lambda_ = self.lambda_ 
        mu = self.loc
        delta = self.scale
        beta = self.asym
        gamma = self.gamma
        return mu+(delta*beta*ss.kn(lambda_+1, delta*gamma))/(gamma*ss.kn(lambda_, delta*gamma))

    def var(self):
        """
        Returns: Variance of the Generalized Hyperbolic distribution.
        """
        λ = self.lambda_ 
        δ = self.scale
        β = self.asym
        γ = self.gamma
        return (δ*ss.kn(λ+1, δ*γ))/(γ*ss.kn(λ, δ*γ))+(pow(β,2)*pow(δ,2))/(pow(γ,2))*(ss.kn(λ+2, δ*γ)/ss.kn(λ, δ*γ)-pow(ss.kn(λ+1, δ*γ), 2)/pow(ss.kn(λ, δ*γ),2))

    def std(self):
        """
        Returns: Standard deviation of the Generalized Hyperbolic distribution
        """
        return sqrt(self.var())

    def summary(self):
        """
        Returns: Summary statistic regarding the Generalized Hyperbolic distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class GN_V1(Base):
    """
    This class contains methods concerning Generalized Normal Distirbution V1. 
    Args:
    
        loc(float): location parameter
        scale(float): scale parameter
        shape(float): shape parameter
        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2021, January 14). Generalized normal distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 10:50, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Generalized_normal_distribution&oldid=1000235907
    """
    def __init__(self, loc, scale, shape, randvar=0):
        self.loc = loc
        self.scale = scale
        self.shape = shape
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Generalized Normal distribution V1.
        """
        _generator = lambda u, b, a, x: b/(2*a*ss.gamma(1/b))*np.exp(-pow(abs(x-u)/a,b))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.shape, self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Generalized Normal distribution V1.
        """

        _generator = lambda u, b, a, x: 0.5+(np.sign(x-u)/2)*(1/ss.gamma(1/b))*ss.gammainc(1/b, pow(x*a,b))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(self.loc, self.scale, self.shape, i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.loc, self.scale, self.shape, self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Generalized Normal distribution V1 evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def = lambda u, b, a, x: 0.5+(np.sign(x-u)/2)*(1/ss.gamma(1/b))*ss.gammainc(1/b, pow(x*a,b))

        return _cdf_def(self.loc, self.scale, self.shape, x_upper)-_cdf_def(self.loc, self.scale, self.shape, x_lower)

    def mean(self):
        """
        Returns: Mean of the Generalized Normal distribution V1.
        """
        return self.loc

    def median(self):
        """
        Returns: Median of the Generalized Normal distribution V1.
        """
        return self.loc

    def mode(self):
        """
        Returns: Mode of the Generalized Normal distribution V1.
        """
        return self.loc

    def var(self):
        """
        Returns: Variance of the Generalized Normal distribution V1.
        """
        a = self.scale
        b = self.shape
        return (pow(a,2)*ss.gamma(3/b))/ss.gamma(1/b)

    def std(self):
        """
        Returns: Standard deviation of the Generalized Normal distribution V1
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Generalized Normal distribution V1. 
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Generalized Normal distribution V1. 
        """
        b = self.shape
        return (ss.gamma(5/b)*ss.gamma(1/b)/pow(ss.gamma(3/b),2)) - 3
    
    def entropy(self):
        """
        Returns: differential entropy of the Generalized Normal distribution V1.
        """
        a = self.scale
        b = self.shape
        return 1/b - np.log(b/(2*a*ss.gamma(1/b)))


    def summary(self):
        """
        Returns: Summary statistic regarding the Generalized Normal distribution V1
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)

# function for standard normal pdf and cdf
# class GN_V2(Base):
#     """
#     This class contains methods concerning Generalized Normal Distirbution V1. 
#     Args:
    
#         loc(float): location parameter
#         scale(float): scale parameter
#         shape(float): shape parameter
#         randvar(float): random variable

#     Methods:

#         - pdf for probability density function.
#         - cdf for cumulative distribution function.
#         - pvalue for p-values.
#         - mean for evaluating the mean of the distribution.
#         - median for evaluating the median of the distribution.
#         - mode for evaluating the mode of the distribution.
#         - var for evaluating the variance of the distribution.
#         - std for evaluating the standard deviation of the distribution.
#         - skewness for evaluating the skewness of the distribution.
#         - kurtosis for evaluating the kurtosis of the distribution.
#         - entropy for differential entropy of the distribution.
#         - summary for printing the summary statistics of the distribution. 

#     Reference:
#     - Wikipedia contributors. (2021, January 14). Generalized normal distribution. In Wikipedia, The Free Encyclopedia. 
#       Retrieved 10:50, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Generalized_normal_distribution&oldid=1000235907
#     """
#     def __init__(self, loc, scale, shape, randvar=0):
#         self.loc = loc
#         self.scale = scale
#         self.shape = shape
#         self.randvar = randvar

#     def pdf(self,
#             plot=False,
#             interval = 0,
#             threshold=1000,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either probability density evaluation for some point or plot of Generalized Normal distribution V1.
#         """
#         _generator = lambda u, b, a, x: b/(2*a*ss.gamma(1/b))*np.exp(-pow(abs(x-u)/a,b))

#         if plot == True:
#             if interval<0:
#                 raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.loc, self.scale, self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.loc, self.scale, self.shape, self.randvar)

#     def cdf(self,
#             plot=False,
#             threshold=1000,
#             interval=1,
#             xlim=None,
#             ylim=None,
#             xlabel=None,
#             ylabel=None):
#         """
#         Args:
        
#             interval(int): defaults to none. Only necessary for defining plot.
#             threshold(int): defaults to 1000. Defines the sample points in plot.
#             plot(bool): if true, returns plot.
#             xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
#             ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
#             xlabel(string): sets label in x axis. Only relevant when plot is true. 
#             ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
#         Returns: 
#             either cumulative distribution evaluation for some point or plot of Generalized Normal distribution V1.
#         """

#         _generator = lambda u, b, a, x: 0.5+(np.sign(x-u)/2)*(1/ss.gamma(1/b))*ss.gammainc(1/b, pow(x*a,b))

#         if plot == True:
#             x = np.linspace(-interval, interval, int(threshold))
#             y = np.array([_generator(self.loc, self.scale, self.shape, i) for i in x])
#             return super().plot(x, y, xlim, ylim, xlabel, ylabel)
#         return _generator(self.loc, self.scale, self.shape, self.randvar)

#     def pvalue(self, x_lower=0, x_upper=None):
#         """
#         Args:

#             x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
#             x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

#             Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
#             Otherwise, the default random variable is x.

#         Returns:
#             p-value of the Generalized Normal distribution V1 evaluated at some random variable.
#         """
#         if x_upper == None:
#             x_upper = self.randvar
#         if x_lower>x_upper:
#             raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
#         _cdf_def = lambda u, b, a, x: 0.5+(np.sign(x-u)/2)*(1/ss.gamma(1/b))*ss.gammainc(1/b, pow(x*a,b))

#         return _cdf_def(self.loc, self.scale, self.shape, x_upper)-_cdf_def(self.loc, self.scale, self.shape, x_lower)

#     def mean(self):
#         """
#         Returns: Mean of the Generalized Normal distribution V1.
#         """
#         return self.loc

#     def median(self):
#         """
#         Returns: Median of the Generalized Normal distribution V1.
#         """
#         return self.loc

#     def mode(self):
#         """
#         Returns: Mode of the Generalized Normal distribution V1.
#         """
#         return self.loc

#     def var(self):
#         """
#         Returns: Variance of the Generalized Normal distribution V1.
#         """
#         a = self.scale
#         b = self.shape
#         return (pow(a,2)*ss.gamma(3/b))/ss.gamma(1/b)

#     def std(self):
#         """
#         Returns: Standard deviation of the Generalized Normal distribution V1
#         """
#         return sqrt(self.var())

#     def skewness(self):
#         """
#         Returns: Skewness of the Generalized Normal distribution V1. 
#         """
#         return 0

#     def kurtosis(self):
#         """
#         Returns: Kurtosis of the Generalized Normal distribution V1. 
#         """
#         b = self.shape
#         return (ss.gamma(5/b)*ss.gamma(1/b)/pow(ss.gamma(3/b),2)) - 3
    
#     def entropy(self):
#         """
#         Returns: differential entropy of the Generalized Normal distribution V1.
#         """
#         a = self.scale
#         b = self.shape
#         return 1/b - np.log(b/(2*a*ss.gamma(1/b)))


#     def summary(self):
#         """
#         Returns: Summary statistic regarding the Generalized Normal distribution V1
#         """
#         mean = self.mean()
#         median = self.median()
#         mode = self.mode()
#         var = self.var()
#         std = self.std()
#         skewness = self.skewness()
#         kurtosis = self.kurtosis()
#         cstr = " summary statistics "
#         print(cstr.center(40, "="))
#         return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Hyperbolic_secant(Base):
    """
    This class contains methods concerning Hyperbolic secant Distirbution. 
    Args:

        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, November 17). Hyperbolic secant distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 11:01, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Hyperbolic_secant_distribution&oldid=989175763
    """
    def __init__(self, randvar=0):
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Hyperbolic secant distribution.
        """
        _generator = lambda x: 0.5*(1/np.cosh(np.pi/2*x)) # sech by the relationship of sech and cosh, verify result!

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Hyperbolic secant distribution.
        """

        _generator = lambda x: (2/np.pi)*np.arctan(np.exp(np.pi/2*x))

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator( i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Hyperbolic secant distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def = lambda x: (2/np.pi)*np.arctan(np.exp(np.pi/2*x))

        return _cdf_def(x_upper)-_cdf_def(x_lower)

    def mean(self):
        """
        Returns: Mean of the Hyperbolic secant distribution.
        """
        return 0

    def median(self):
        """
        Returns: Median of the Hyperbolic secant distribution.
        """
        return 0

    def mode(self):
        """
        Returns: Mode of the Hyperbolic secantl distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Hyperbolic secantl distribution.
        """
        return 1

    def std(self):
        """
        Returns: Standard deviation of the Hyperbolic secant distribution
        """
        return sqrt(self.var())

    def skewness(self):
        """
        Returns: Skewness of the Hyperbolic secant distribution. 
        """
        return 0

    def kurtosis(self):
        """
        Returns: Kurtosis of the Hyperbolic secant distribution. 
        """
        b = self.shape
        return 2

    def entropy(self):
        """
        Returns: (approximation) differential entropy of the Hyperbolic secant distribution.
        """
        a = self.scale
        b = self.shape
        return 1.16624


    def summary(self):
        """
        Returns: Summary statistic regarding the Hyperbolic secant distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)


class Slash(Base):
    """
    This class contains methods concerning Slash Distirbution. 
    Args:

        randvar(float): random variable

    Methods:

        - pdf for probability density function.
        - cdf for cumulative distribution function.
        - pvalue for p-values.
        - mean for evaluating the mean of the distribution.
        - median for evaluating the median of the distribution.
        - mode for evaluating the mode of the distribution.
        - var for evaluating the variance of the distribution.
        - std for evaluating the standard deviation of the distribution.
        - skewness for evaluating the skewness of the distribution.
        - kurtosis for evaluating the kurtosis of the distribution.
        - entropy for differential entropy of the distribution.
        - summary for printing the summary statistics of the distribution. 

    Reference:
    - Wikipedia contributors. (2020, February 18). Slash distribution. In Wikipedia, The Free Encyclopedia. 
    Retrieved 11:30, January 19, 2021, from https://en.wikipedia.org/w/index.php?title=Slash_distribution&oldid=941483761
    """
    def __init__(self, randvar=0):
        self.randvar = randvar

    def pdf(self,
            plot=False,
            interval = 0,
            threshold=1000,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either probability density evaluation for some point or plot of Slash distribution.
        """
        _generator = lambda x: (super().std_normal_pdf(0) - super().std_normal_pdf(0))/ pow(x,2) if x!=0 else 1/(2*sqrt(2*np.pi))

        if plot == True:
            if interval<0:
                raise ValueError('interval should not be less then 0. Entered value: {}'.format(interval))
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator(i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar)

    def cdf(self,
            plot=False,
            threshold=1000,
            interval=1,
            xlim=None,
            ylim=None,
            xlabel=None,
            ylabel=None):
        """
        Args:
        
            interval(int): defaults to none. Only necessary for defining plot.
            threshold(int): defaults to 1000. Defines the sample points in plot.
            plot(bool): if true, returns plot.
            xlim(float): sets x axis ∈ [-xlim, xlim]. Only relevant when plot is true.
            ylim(float): sets y axis ∈[0,ylim]. Only relevant when plot is true. 
            xlabel(string): sets label in x axis. Only relevant when plot is true. 
            ylabel(string): sets label in y axis. Only relevant when plot is true. 

        
        Returns: 
            either cumulative distribution evaluation for some point or plot of Slash distribution.
        """

        _generator = lambda x: super().std_normal_cdf(x)-(super().std_normal_pdf(0)-super().std_normal_pdf(x))/x if x!=0 else 0.5

        if plot == True:
            x = np.linspace(-interval, interval, int(threshold))
            y = np.array([_generator( i) for i in x])
            return super().plot(x, y, xlim, ylim, xlabel, ylabel)
        return _generator(self.randvar)

    def pvalue(self, x_lower=0, x_upper=None):
        """
        Args:

            x_lower(float): defaults to 0. Defines the lower value of the distribution. Optional.
            x_upper(float): defaults to None. If not defined defaults to random variable x. Optional.

            Note: definition of x_lower and x_upper are only relevant when probability is between two random variables.
            Otherwise, the default random variable is x.

        Returns:
            p-value of the Slash distribution evaluated at some random variable.
        """
        if x_upper == None:
            x_upper = self.randvar
        if x_lower>x_upper:
            raise Exception('lower bound should be less than upper bound. Entered values: x_lower:{} x_upper:{}'.format(x_lower, x_upper))
        
        _cdf_def = lambda x: (2/np.pi)*np.arctan(np.exp(np.pi/2*x))

        return _cdf_def(x_upper)-_cdf_def(x_lower)

    def mean(self):
        """
        Returns: Mean of the Slash distribution.
        """
        return "Does not exist."

    def median(self):
        """
        Returns: Median of the Slash distribution.
        """
        return 0

    def mode(self):
        """
        Returns: Mode of the Slash distribution.
        """
        return 0

    def var(self):
        """
        Returns: Variance of the Slash distribution.
        """
        return "Does not exist."

    def std(self):
        """
        Returns: Standard deviation of the Slash distribution
        """
        return "Does not exist."

    def skewness(self):
        """
        Returns: Skewness of the Slash distribution. 
        """
        return "Does not exist."

    def kurtosis(self):
        """
        Returns: Kurtosis of the Slash distribution. 
        """
        return "Does not exist."


    def summary(self):
        """
        Returns: Summary statistic regarding the Slash distribution
        """
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.var()
        std = self.std()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = " summary statistics "
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode, "\nvar: ", var, "\nstd: ", std, "\nskewness: ", skewness, "\nkurtosis: ", kurtosis)
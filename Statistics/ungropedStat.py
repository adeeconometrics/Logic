try:
    import numpy as np
    # import scipy as sci
    # import scipy.special as ss
    # import sympy as sp
except Exception as e:
    print("some modules are missing {}".format(e))

# todo: assesing mode for bimodality and multimodality

class ungroupedStatistics:
    '''
    The ungroupedStatistics object contains methods for ungrouped statistics summary that describes the distribution.

    Args: data(list): raw data set
    '''
    def __init__(self, data):
        self.data = data

    def mean(self):
        '''
        Reutrns: 
            arithmetic mean
        '''
        return np.mean(self.data)

    def median(self):
        '''
        Returns: ungrouped median
        '''
        data = self.data
        if (len(data) % 2 == 0):
            for i in data:
                for j in data:
                    if (data[i] < data[j]):
                        temp = data[i]
                        data[i] = data[j]
                        data[j] = temp
            mid = int(len(data) / 2)

            return (data[mid] + data[mid + 1]) / 2

        mid = int(len(data) / 2)
        return data[mid + 1]

    def mode(self):  # does not account for bimodal and multimodal cases
        data_set = list(set(self.data))
        frequency = [(self.data.count(i), i) for i in data_set]
        return max(frequency)

    def range(self):
        '''
        Returns: range = max - min
        '''
        return max(self.data) - min(self.data)

    def pop_variance(self):
        '''
        Returns: population variance
        '''
        mean = self.mean()
        size = len(self.data)
        return (1 / size) * sum([(x - mean)**2 for x in self.data])

    def pop_std(self):
        '''
        Returns:  population standard deviation
        '''
        mean = self.mean()
        var = self.pop_variance()
        return np.sqrt(var)

    def samp_variance(self):
        '''
        Returns: sample variance
        '''
        mean = self.mean()
        size = len(self.data)
        return (1 / (size - 1)) * sum([(x - mean)**2 for x in self.data])

    def samp_std(self):
        '''
        Returns: sample standard deviation
        '''
        mean = self.mean()
        var = self.samp_variance()
        return np.sqrt(var)

    def skewness(self, mean=None, median=None, samp_std=None, return_sk=False):
        '''
        Args:

            mean (float): if the mean value is given. This is optional
            median (float): if the median value is given. This is optional
            samp_std (float): if the sample standard deviation given. This is optional
            return_sk (bool): should the skewness value be returned or not. This is optional.

        Returns: skewness: either description or value
        '''
        if mean == None:
            mean = self.mean()
        if median == None:
            median = self.median()
        if samp_std == None:
            samp_std = self.samp_std()

        sk = (3 * (mean - median) / samp_std)

        if return_sk == True:
            return sk

        if (sk == 0):
            print(sk, "- the distribution is normal")
        if (sk < 0):
            print(sk, " - the distribution is skewed to the left.")
        if (sk > 0):
            print(sk, " -  the distribution is skewed to the right.")

    def kurtosis(self,
                 samp_mean=None,
                 samp_size=None,
                 samp_std=None,
                 return_ku=False):
        '''
        Args:

            samp_mean (float): if the sample mean is given. This is optional.
            samp_size (int): if the sample size is given. This is optional
            samp_std (float): is the sample standard deviation is given. This is optional.
            return_ku (bool): defaults to False (returns 
            description of ku),returns 
            ku value is True
        
        Returns: Kurtosis: either description of value
        '''
        if samp_mean == None:
            samp_mean = self.mean()
        if samp_size == None:
            samp_size = len(data)
        if samp_std == None:
            samp_std = np.std(data)

        ku = np.sum(np.array([(x - samp_mean)**4
                              for x in data])) / (samp_size * samp_std**4)

        if return_ku == True:
            return ku

        if (ku == 3):
            print("the distirbution is mesokurtic")
        if (ku < 3):
            print("the distribution is leptokurtic")
        if (ku > 3):
            print("the distribution is platykurtic")

    def print_summary(self):
        '''
        Returns: prints summary statistics.
        '''
        mean = self.mean()
        median = self.median()
        mode = self.mode()
        var = self.samp_variance()
        skewness = self.skewness()
        kurtosis = self.kurtosis()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
                     "\nsample var: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
                     kurtosis)


class centralTendency:
    '''
    The centralTendency object contains methods for cental tendencies i.e. mean, median, and mode.

    Args: data(list): raw data set
    '''
    def __init__(self, data):
        self.data = data

    def arithmetic_mean(self):
        '''
        Returns: arithmetic mean
        '''
        return np.sum(self.data) / len(self.data)

    def geometric_mean(self):
        '''
        Returns: geometric mean
        '''
        return np.power(np.prod(self.data), 1 / len(self.data))

    def harmonic_mean(self):
        '''
        Returns: harmonic mean
        '''
        return len(self.data) * np.power(
            np.sum([1 / self.data[i] for i in range(0, len(self.data))]), -1)

    def root_mean_square(self):
        '''
        Returns: root mean square or quadratic mean.
        '''
        return np.sqrt(
            (1 / len(self.data)) *
            np.sum(np.power(data[i], 2) for i in range(0, len(self.data))))

    def weigthed_mean(self, weights):
        '''
        Args:

            weights(list of integers): necessary.
        Returns: weigthed mean.
        '''
        if (len(self.data) == len(weights)):
            return (np.sum(
                [self.data[i] * weights[i]
                 for i in range(0, len(self.data))])) / (np.sum(weights))
        else:
            print("error: data and weights should have the same size")

    def interquartile_mean(self):
        '''
        Returns: interquartile mean
        '''
        scale = 1 / len(self.data)
        upper_limit = np.ceil(3 / 4 * len(self.data))
        lower_limit = np.floor(len(self.data) / 4 + 1)
        return scale * np.sum(
            [self.data[i] for i in range(lower_limit, upper_limit)])

    def generalized_mean(self, m):
        '''
        Args:

            m (float)
        Returns:
        
            root mean square - when m = 2
            arithmetic mean - when m=1
            geometric mean - when m approaches to 0
            harmonic mean - when m = -1
        '''
        return np.power((1 / len(self.data)) * np.sum(self.data), 1 / m)

    def median(self):
        '''
        Returns: median
        '''
        data = self.data
        if (len(data) % 2 == 0):
            for i in data:
                for j in data:
                    if (data[i] < data[j]):
                        temp = data[i]
                        data[i] = data[j]
                        data[j] = temp
            mid = int(len(data) / 2)

            return (data[mid] + data[mid + 1]) / 2

        mid = int(len(data) / 2)
        return data[mid + 1]

    def mode(self):  # does not account for bimodal and multimodal cases
        data_set = list(set(self.data))
        frequency = [(self.data.count(i), i) for i in data_set]
        return max(frequency)


class Dispersion:
    def __init__(self, data):
        self.data = data

    def range(self):
        '''
        Returns: range = max - min 
        '''
        return max(self.data) - min(self.data)

    def pop_variance(self):
        '''
        Returns: population variance
        '''
        mean = self.mean()
        size = len(self.data)
        return (1 / size) * sum([(x - mean)**2 for x in self.data])

    def pop_std(self):
        '''
        Returns: population standard deviation
        '''
        mean = self.mean()
        var = self.pop_variance()
        return np.sqrt(var)

    def samp_variance(self):
        '''
        Returns: sample variance
        '''
        mean = self.mean()
        size = len(self.data)
        return (1 / (size - 1)) * sum([(x - mean)**2 for x in self.data])

    def samp_std(self):
        '''
        Returns: sample standard deviation
        '''
        mean = self.mean()
        var = self.samp_variance()
        return np.sqrt(var)

    def coef_variation(self):
        '''
        Returns: coefficient of variation
        '''
        std = np.std(self.data)
        samp_mean = np.mean(self.data)
        return std / samp_mean

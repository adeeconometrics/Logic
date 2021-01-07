try:
    import numpy as np
    from math import sqrt

except Exception as e:
    print("some modules are missing {}".format(e))

# RENAME THIS FILE SOON

'''
test statistic is used in hypotheses tests. This module contains a collection of
test statistic implementations. The modules for distirbutions shall interact with 
this module for computing p-values. Functions must be independent of their own, while
classes can interact with their subroutines, avoid unnecessary dependencies.
'''

def z_transformation(samp_mean, pop_mean, std):
    '''
    a function for transforming all normally distributed variables to standard distribution of variables (mean=0, std=1).
    
    Args:

        samp_mean(float): sample mean
        pop_mean(float): population mean
        std (float): standard deviation
    '''
    return (samp_mean - pop_mean) / std


def z_score(samp_mean, pop_mean, std):
    '''
    Args:

        samp_mean (float): sample mean 
        pop_mean (float): population mean
        std (float): population standard deviation

    Assumptions:
    
        - test statistic should follow a normal distribution. If the variation is strongly non-normal, a z-test should not be used.
        - nuisance parameters (e.g. std in one-sample location test) should be known or estimated with high accuracy
    
    Returns:
        z-score
    '''
    return (samp_mean - pop_mean) / std


def z_score_smean(samp_mean, pop_mean, std, n):
    '''
    Args:

        n (int): sample size
        samp_mean (float): sample mean 
        pop_mean (float): population mean
        std (float): standard deviation

    assumptions:
    
        - test statistic should follow a normal distribution. If the variation is strongly non-normal, a z-test should not be used.
        - nuisance parameters (e.g. std in one-sample location test) should be known or estimated with high accuracy
        - the mean of the population is known 
        - n>=30 of when the population is normally distributed std is known

    Returns: 
        test value based on z-statistic for single mean
    '''
    return (samp_mean - pop_mean) / (std / np.sqrt(n))


def z_score_dmean(samp_mean1,
                  samp_mean2,
                  std1,
                  std2,
                  n1,
                  n2,
                  pop_mean1=None,
                  pop_mean2=None):
    '''
    Args:

        samp_mean1(float): sample mean of the first distirbution.
        samp_mean2(float): sample mean of the second distribution.
        std1(float): standard deviation of first distribution.
        std2(float): standard deviation of second distribution.
        n1(int): sample size of first distribution.
        n2(int): sample size of second distribution.
        pop_mean1(float): population mean of first distribution. This is optional.
        pop_mean2(float): population mean of second distribution. This is optional.
    
    assumptions:
    
        - test statistic should follow a normal distribution. If the variation is strongly non-normal, a z-test should not be used.
        - nuisance parameters (e.g. std in one-sample location test) should be known or estimated with high accuracy
        - the mean of the population is known 
        - n>=30 of when the population is normally distributed std is known
    
    Returns:
        test value based on z-statistic for two means
    '''
    if pop_mean1 is None and pop_mean2 is None:
        return (samp_mean1 - samp_mean2) / (np.power((np.power(std1, 2) / n1) +
                                                     (np.power(std2, 2) / n2)))
    return ((samp_mean1 - samp_mean2) -
            (pop_mean1 - pop_mean2)) / (np.power((np.power(std1, 2) / n1) +
                                                 (np.power(std2, 2) / n2)))


def t_test(samp_mean, pop_mean, samp_std, n):
    '''
    Args:

        samp_mean(float): sample mean.
        pop_mean(float): population mean.
        samp_std(float): sample standard deviation
        n(int): sample size
    
    asusmptions:
        - sample size is small n<=30
        - population standard deviation is not known
        - the population is normally or approximately normally distributed

    Returns:
        test value based on t-statistic
    '''
    return (samp_mean - pop_mean) / (samp_std / np.sqrt(n))


def t_test_dmean_uneq(samp_mean1,
                      samp_mean2,
                      pop_mean1=None,
                      pop_mean2=None,
                      std1,
                      std2,
                      n1,
                      n2):
    '''
    also known as Welch's t-test.
    Args:

        samp_mean1(float): sample mean of the first distirbution.
        samp_mean2(float): sample mean of the second distribution.
        std1(float): standard deviation of first distribution.
        std2(float): standard deviation of second distribution.
        n1(int): sample size of first distribution.
        n2(int): sample size of second distribution.
        pop_mean1(float): population mean of first distribution. This is optional.
        pop_mean2(float): population mean of second distribution. This is optional.

    asusmptions:
        - sample size is small n<=30
        - population standard deviation is not known
        - the population is normally or approximately normally distributed
    
    Returns: 
        test value based on t-statistic between two means for unequal variance 
    '''
    if (pop_mean1 == None and pop_mean2 == None):
        return (samp_mean1 - samp_mean2) / (np.sqrt((std1**2 / n1) +
                                                    (std2**2 / n2)))
    return ((samp_mean1 - samp_mean2) -
            (pop_mean1 - pop_mean2)) / (np.sqrt((std1**2 / n1) +
                                                (std2**2 / n2)))


def t_test_dmean_eq(samp_mean1,
                    samp_mean2,
                    pop_mean1=None,
                    pop_mean2=None,
                    std1,
                    std2,
                    n1,
                    n2):
    '''
    Args:

        samp_mean1(float): sample mean of the first distirbution.
        samp_mean2(float): sample mean of the second distribution.
        std1(float): standard deviation of first distribution.
        std2(float): standard deviation of second distribution.
        n1(int): sample size of first distribution.
        n2(int): sample size of second distribution.
        pop_mean1(float): population mean of first distribution. This is optional.
        pop_mean2(float): population mean of second distribution. This is optional.

    assumptions:
    
        - sample size is small n<=30
        - population standard deviation is not known
        - the population is normally or approximately normally distributed
    
    Returns: 
        test value based on t-statistic between two means for equal variance. 
    '''
    if (pop_mean1 != None and pop_mean2 !=
            None):  # is there really a case where pop_mean1 is only given?
        return ((samp_mean1 - samp_mean2) -
                (pop_mean1 - pop_mean2)) / (np.sqrt(((n1 - 1) * std1**2) +
                                                    ((n2 - 1) * std2**2) /
                                                    (n1 + n2 - 2)) *
                                            np.sqrt(1 / n1 + 1 / n2))
    return (samp_mean1 - samp_mean2) / (np.sqrt(((n1 - 1) * std1**2) + (
        (n2 - 1) * std2**2) / (n1 + n2 - 2)) * np.sqrt(1 / n1 + 1 / n2))

def paired_t_test(data_set1, data_set2): # test functionality
    '''
    this is appropriate for compairing two samples where it is impossible to control important variables.
    Args:

        data_set1(list): contains pre-test data set
        data_set2(list): contains post-test data set
    
    assumptions:
    
        -  standard deviation is unknown
        -  len(data_set1)==len(data_set2)
    Returns:
        test value based on t-statistic for paired samples. 
    '''
    if len(data_set1)!=len(data_set2): 
        return print("data sets must have equal size.")
    sum_diff = sum([(x-y) for x in data_set1 for y in data_set2])
    ss_diff = sum([(x-y)**2 for x in data_set1 for y in data_set2])
    size = len(data_set1)
    df = size-1
    t_value = (sum_diff/size)/np.sqrt((ss_diff-(sum_diff**2)/size)/(df*size)
    return t_value

def regression_t_test():
    pass

def chi_squared_test():
    pass 

def chi_squared_test_independence():
    pass

def chi_squared_test_goodnessfit():
    pass

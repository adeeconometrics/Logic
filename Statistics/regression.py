import numpy as np
import scipy.special as ss
from math import sqrt
from distributions import univariate
class generalMethods:
    '''
    Base class architecture of generalMethods Data structure: Dictionaries of independent and dependent variables.
    '''
    def __init__(self,
                 independent=None,
                 dependent=None,
                 covariance=None,
                 adjust=True):
        if isinstance(dependent, (dict, type(None))) and isinstance(
                independent,
            (dict, type(None))) and isinstance(covariance,
                                               (dict, type(None))) is False:
            if isinstance(dependent, dict) == False:
                raise TypeError('dependent variable should be a dictionary.')
            if isinstance(independent, dict) == False:
                raise TypeError('independent variable should be a dictionary.')
            raise TypeError('covariance variable should be a dictionary.')

        # this can be improved to pre-compare lengths
        self.factor = self.temp_max = 0
        self.dependent = dependent
        self.independent = independent
        self.covariance = covariance

        if dependent is not None:
            self.keys_d = list(dependent.keys())
            self.factor += len(self.keys_d)

        if independent is not None:
            self.keys_ind = list(independent.keys())
            self.factor += len(self.keys_ind)

        if covariance is not None:
            self.keys_cov = list(covariance.keys())
            self.factor += len(self.keys_cov)

        if adjust == False:
            pass
        self.adjust_set()

    def adjust_set(self):
        _max_d = _max_ind = _max_cov = 0
        # find max
        if self.dependent is not None:
            for var in range(0, len(self.keys_d)):
                if _max_d < len(self.dependent[self.keys_d[var]]):
                    _max_d = len(self.dependent[self.keys_d[var]])

        if self.independent is not None:
            for var in range(0, len(self.keys_ind)):
                if _max_ind < len(self.independent[self.keys_ind[var]]):
                    _max_ind = len(self.independent[self.keys_ind[var]])

        if self.covariance is not None:
            for var in range(0, len(self.keys_cov)):
                if _max_cov < len(self.covariance[self.keys_cov[var]]):
                    _max_cov = len(self.covariance[self.keys_cov[var]])
        # update values
        self.temp_max = max([_max_cov, _max_d, _max_ind])
        _max_d = _max_ind = _max_cov = self.temp_max

        if self.dependent is not None:
            for var in range(0, len(self.keys_d)):
                if _max_d - len(self.dependent[self.keys_d[var]]) != 0:
                    diff = _max_d - len(self.dependent[self.keys_d[var]])
                    update_d = self.dependent[self.keys_d[var]] + [0] * diff
                    self.dependent.update([(self.keys_d[var], update_d)])

        if self.independent is not None:
            for var in range(0, len(self.keys_ind)):
                if _max_ind - len(self.independent[self.keys_ind[var]]) != 0:
                    diff = _max_ind - len(self.independent[self.keys_ind[var]])
                    update_ind = self.independent[
                        self.keys_ind[var]] + [0] * diff
                    self.independent.update([(self.keys_ind[var], update_ind)])

        if self.covariance is not None:
            for var in range(0, len(self.keys_cov)):
                if _max_cov - len(self.covariance[self.keys_cov[var]]) != 0:
                    diff = _max_cov - len(self.covariance[self.keys_cov[var]])
                    update_cov = self.covariance[
                        self.keys_cov[var]] + [0] * diff
                    self.covariance.update([(self.keys_cov[var], update_cov)])

    def t_statistic(self, cv, df, ci=None):
        '''
        Args:

            - cv(float): critical vaue
            - df(int): degrees of freedom
            - ci(int | [0,100]): Optional. confidence interval
        
        Returns: p-value drawn from student's T distribution
        ''' # check this code 
        p_value = univariate.T_distribution(df, cv).p_value()
        return p_value

    def f_statistic(self, cv, df1, df2, ci=None):
        '''
        Args:

            - cv(float): critical vaue
            - df1(int): degrees of freedom
            - df2(int): degrees of freedom
            - ci(int | [0,100]): Optional. confidence interval
        
        Returns: p-value drawn from F distribution
        '''# check this code 
        p_value = univariate.F_distribution(cv,df1,df2).p_value()
        return p_value

    def wrap_data(self, on=**kwargs):
        '''
        read csv file, excel, and convert to numpy array
        '''
        pass

    def least_squares(self):
        pass

    def linear_least_sq(self):
        pass

    def non_linear_least_sq(self):
        pass

    def weighted_least_sq(self):
        pass

    def least_absolute_dev(self):
        pass

    def fit_curve(self):
        pass

    def f_test(self):
        pass

    def t_test(self):
        pass

    def cooefficient_determination(self):
        pass

    def multiple_correlation(self):
        pass

    def correlation(self):
        '''
        Returns: Pearson's correlation coefficient
        '''
        pass


class Diagnostics:
    pass


class GLM(generalMethods):
    def __init__(self):
        pass

    def logistic_ordered(self):
        pass

    def logistic_multinomial(self):
        pass

    def probit_ordered(self):
        pass

    def probit_multinomial(self):
        pass

    def poisson(self):
        pass

    def max_likelihood(self):
        pass

    pass


class OLS(generalMethods):
    pass



class correlation:
    '''
    This class contains implementation concerning correlation analysis which is used to measure the
    strength and direction between two variables. This class offers two methods of correlation:
    the Pearson product moment correlation and the Spearman rank order correlation. 

    Methods:
    - p_value - for t-statistic's p-value
    - pearson_correlation - for Pearson product moment correlation
    - spearman_correlation - for Spearman rank order correlation 

    Reference: 
    - Minitab (2019) Correlation Analysis. Retrieved at: 
    https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/how-to/correlation/methods-and-formulas/
    '''

    def __init__(self):
        pass

    def p_value(self):
        pass

    def pearson_correlation(self):
        pass 

    def spearman_correlation(self):
        pass 


class simpleLinearRegression(generalMethods):
    
    def __init__(self, data_x, data_y):
        if isinstance(data_x, dict) and isinstance(data_y, dict) == False:
            raise TypeError('Independent and dependent variables must be a dictionary.')
        if (len(data_x.keys())>1) and (len(data_y)>1):
            raise Exception('There must only one dependent and one independent variable. This condition is unsatisfied.')
        super().__init__(independent=data_x, dependent=data_y)

    def correlation(self):
        '''
        Returns: Pearson's correlation coefficient
        '''
        key_x = self.keys_ind[0]
        key_y = self.keys_d[0]
        mean_x = sum(self.independent[key_x])/len(self.independent[key_x])
        mean_y = sum(self.dependent[key_y])/len(self.dependent[key_y])
        temp_u = sum([x-mean_x for x in self.independent[key_x]])*sum([y-mean_y for y in self.dependent[key_y]])
        temp_l = sqrt(sum([(x-mean_x)**2 for x in self.independent[key_x]]))*sqrt([(y-mean_y**2 for y in self.dependent[key_y])])
        return temp_u/temp_l

    def r_square(self, adjusted=False):
        '''
        Args:

            - adjusted(bool): defaults to False. If true, returns adjusted r-square.
        Returns: r-square or adjusted r-square.
        '''
        sample_mean = sum(self.dependent[self.keys_d[0]])/len(self.dependent[self.keys_d[0]])
        # rss = sum([])
        # tss = sum([])
        pass

    def std_err(self):
        '''
        Returns: standard error estimate
        '''
        pass

    def a_coef(self):
        '''
        Returns: a coefficient
        '''
        pass

    def b_coef(self):
        '''
        Returns: b coefficient 
        '''
        pass

    def print_summary(self):
        '''
        Returns: prints summary statistic. 
        '''
        pass
    pass


class trendEstimation(generalMethods):
    pass


class ridgeRegression(generalMethods):
    pass


class polynomialRegression(generalMethods):
    pass


class segmentedRegression(generalMethods):
    pass


class nonLinearRegression(generalMethods):
    pass
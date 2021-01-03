import numpy as np
import scipy.special as ss

class generalMethods:
    '''
    Base class architecture of generalMethods Data structure: Dictionaries of independent and dependent variables.
    '''
    def __init__(self, independent, dependent, adjust=True):
        self.dependent = dependent
        self.independent = independent
        self.keys_d = list(dependent.keys())
        self.keys_ind = list(independent.keys())
        self.factor = len(self.keys_d) + len(self.keys_ind)

        if adjust == False:
            pass
        self.adjust_set()

    def adjust_set(self):
        self.max_len_d = self.max_len_ind = 0
        for var in range(0, len(self.keys_d)):
            # find max lenght in dependent dict
            if self.max_len_d < len(self.dependent[self.keys_d[var]]):
                self.max_len_d = len(self.dependent[self.keys_d[var]])

        for var in range(0, len(self.keys_d)):
            # fill 0
            if self.max_len_d - len(self.dependent[self.keys_d[var]]) != 0:
                diff = self.max_len_d - len(self.dependent[self.keys_d[var]])
                update_d = self.dependent[self.keys_d[var]] + [0] * diff

                self.dependent.update([(self.keys_d[var], update_d)])

        for var in range(0, len(self.keys_ind)):
            # find max lenght in independent dict
            if self.max_len_ind < len(self.independent[self.keys_ind[var]]):
                self.max_len_ind = len(self.independent[self.keys_ind[var]])

        for var in range(0, len(self.keys_ind)):
            # fill 0
            if self.max_len_ind - len(
                    self.independent[self.keys_ind[var]]) != 0:
                diff = self.max_len_ind - len(
                    self.independent[self.keys_ind[var]])
                update_ind = self.independent[self.keys_ind[var]] + [0] * diff

                self.independent.update([(self.keys_ind[var], update_ind)])

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


class General_generalMethods(generalMethods):
    pass


class simpleLinearRegression(generalMethods):
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
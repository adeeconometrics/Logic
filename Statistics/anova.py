try:
    import numpy as np
    # import scipy as sci
    import scipy.special as ss
    import math as m
    # import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))


class BaseAnova:
    '''
    Base class architecture of ANOVA Data structure: Dictionaries of independent and dependent variables.
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


class Anova1(BaseAnova):
    '''
    This class contains methods concerning One-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.
    '''
    def __init__(self, independent, dependent, adjust=True):
        super(Anova1, self).__init__(independent, dependent, adjust)
        # test dataset if it follows the fundamental assumtption with one-way ANOVA

    def sum_squares(self):
        '''
        it is the sum of squares of the deviations from the means.
        '''
        mean_obs = np.mean(self.data, axis=0)
        mean_all = np.mean(self.data)
        n_i = [len(self.data[i] for i in range(0, len(self.data)))]
        return sum([n * (mean_obs - mean_all)**2 for n in n_i])

    def mean_sq(self):
        '''
        it is the kind of "average variations" and is found by dividing the variation by the degrees of freedom. 
        '''
        ss_factor = self.sum_squares()

        # return ms_factor

    def mean_sq_err(self):
        pass

    def between_group_var(self):
        pass

    def within_group_var(self):
        grand_mean = np.mean(self.data)

        pass

    def f_val(self):
        pass

    def p_value(self):
        pass

    def residuals(self):
        pass

    def r_square(self):
        pass

    def adjusted_r_square(self):
        pass

    def print_summary(self):
        sum_squares = self.sum_squares()
        mean_sq = self.mean_sq()
        mean_sq_err = self.mean_sq_err()
        between_group_var = self.between_group_var()
        within_group_var = self.within_group_var()
        f_val = self.f_val()
        p_val = self.p_val()
        r_sq = self.r_square()
        adj_r = self.adjusted_r_square()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        # return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
        #              "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
        #              kurtosis)

    pass


class Anova2(BaseAnova):
    '''
    This class contains methods concerning two-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.
    '''
    def __init__(self, independent, dependent, adjust=True):
        super(Anova2, self).__init__(independent, dependent, adjust)

    def sum_squares(self):
        '''
        it is the sum of squares of the deviations from the means.
        '''
        mean_obs = np.mean(self.data, axis=0)
        mean_all = np.mean(self.data)
        n_i = [len(self.data[i] for i in range(0, len(self.data)))]
        return sum([n * (mean_obs - mean_all)**2 for n in n_i])

    def mean_sq(self):
        '''
        it is the kind of "average variations" and is found by dividing the variation by the degrees of freedom. 
        '''
        ss_factor = self.sum_squares()

        # return ms_factor

    def mean_sq_err(self):
        pass

    def between_group_var(self):
        pass

    def within_group_var(self):
        grand_mean = np.mean(self.data)

        pass

    def f_val(self):
        pass

    def p_value(self):
        pass

    def residuals(self):
        pass

    def r_square(self):
        pass

    def adjusted_r_square(self):
        pass

    def print_summary(self):
        pass

    pass


class Manova(BaseAnova):
    def __init__(self, independent, dependent, adjust=True):
        super(Manova, self).__init__(independent, dependent, adjust)

    pass


class Ancova(BaseAnova):
    def __init__(self, independent, dependent, adjust=True):
        super(Ancova, self).__init__(independent, dependent, adjust)

    pass


class Mancova(BaseAnova):
    def __init__(self, independent, dependent, adjust=True):
        super(Mancova, self).__init__(independent, dependent, adjust)

    pass
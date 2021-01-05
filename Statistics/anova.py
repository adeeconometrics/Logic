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


class PostHoc(BaseAnova):
    '''
    This class covers the methods for post hoc analyses with ANOVA.
    '''
    pass


class Anova1(BaseAnova):
    '''
    This class contains methods concerning One-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.

    One-way ANOVA is used when you have a categorical factor and a continuous response.
    It is ideal for determining whether the means of two or more groups differ and for obtaining
    a range of values for the difference between the means for each pair of the group(1). Also, it is 
    important to realize that one-way ANOVA is an omnibus test statistic which cannot tell you which
    specific groups were statistically signnificantly different from each other, rather it only tells you
    that at least two groups were different(2).

    Assumptions:

        - dependent variable are continuous (e.g. ratio and interval).
        - independent variable must be categorical (nominal). Must have two or more.
        - no significant outliers.
        - there needs to be homogeneity of variances.
        - dependent variable should be approximately normally distirbuted for each combination of the groups of 
        two independent variables. 
        - independence of observation i.e. there is no relationship between the observation in each group or between
        the groups themselves

    This class provides you with a set of methods for one-way ANOVA. Note that assessing the data is not provided in this class.

    Args:

        - independent (dict{string:list}): is a key-value pair that matches the category and groups. 
    
    Methods:

        - sum_squares
        - mean_sq
        - mean_sq_err
        - between_group_var
        - within_group_var
        - f_valueue
        - p_value
        - residuals
        - r_square
        - adjusted_r_sq
        - print_summary

    References:

        (1) Minitab (2019). Overview for One-Way ANOVA. Retrieved at: 
        https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/anova/how-to/one-way-anova/before-you-start/overview/

        (2) Laerd Statistics (2018). One-way ANOVA in SPSS Statistics. Retrieved at: 
        https://statistics.laerd.com/spss-tutorials/one-way-anova-using-spss-statistics.php

        (3) Matlab & Simulink (n.d.). One-Way ANOVA. https://www.mathworks.com/help/stats/one-way-anova.html
    '''
    def __init__(self, independent, adjust=True):
        super(Anova1, self).__init__(independent=independent, adjust=adjust)

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
        '''
        Returns: mean square error for One-way ANOVA.
        '''
        pass

    def between_group_var(self):
        '''
        Returns: between group variance for One-way ANOVA.
        '''
        pass

    def within_group_var(self):
        '''
        Returns: within group variance for One-way ANOVA.
        '''
        grand_mean = np.mean(self.data)

        pass

    def f_value(self):
        '''
        Returns: f-value for One-way ANOVA.
        '''
        pass

    def p_value(self):
        '''
        Returns: p value for One-way ANOVA.
        '''
        pass

    def residuals(self):
        '''
        Returns: sum of squares residuals for One-way ANOVA.
        '''
        pass

    def r_square(self):
        '''
        Returns: r square for One-way ANOVA.
        '''
        pass

    def adjusted_r_sq(self):
        '''
        Returns: adjusted r square for One-way ANOVA.
        '''
        pass

    def print_summary(self):
        '''
        Prints summary statistics for One-way ANOVA.
        '''
        sum_squares = self.sum_squares()
        mean_sq = self.mean_sq()
        mean_sq_err = self.mean_sq_err()
        between_group_var = self.between_group_var()
        within_group_var = self.within_group_var()
        f_value = self.f_value()
        p_val = self.p_val()
        r_sq = self.r_square()
        adj_r = self.adjusted_r_sq()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        # return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
        #              "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
        #              kurtosis)

    pass


class Anova2(BaseAnova):
    # include model assumptions, when is it appropriately used.
    '''
    This class contains methods concerning two-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.

    Two-way ANOVA is used to determine whether group means are different when you have two categorical factors.

    Assumptions:

        - dependent variable are continuous (e.g. ratio and interval).
        - two independent variable must be categorical (nominal). Independent variables should each consist
         of two or more categorical, independent groups.  
        - no significant outliers.
        - there needs to be homogeneity of variance for each combination of the gorups of the two independent variables.
        - dependent variable should be approximately normally distirbuted for each combination of the groups of 
        two independent variables. 
        - independence of observation i.e. there is no relationship between the observation in each group or between
        the groups themselves
    
    This class provides you with a set of methods for two-way ANOVA. Note that assessing the data is not provided in this class.

    Args:
    
    Methods:

        - sum_squares
        - mean_sq
        - mean_sq_err
        - between_group_var
        - within_group_var
        - f_valueue
        - p_value
        - residuals
        - r_square
        - adjusted_r_sq
        - print_summary

    References:

        (1) Laerd Statistics (2018). Two-way ANOVA in SPSS Statistics. Retrieved at: 
        https://statistics.laerd.com/spss-tutorials/two-way-anova-using-spss-statistics.php
        (2) MathWorks [Matlab] (n.d.). Two-way ANOVA. Retrieved at: 
        https://www.mathworks.com/help/stats/two-way-anova.html?searchHighlight=two%20way%20anova&s_tid=srchtitle
        (3) Minitab (2019). Overview for Two-Way ANOVA. Retrieved at: 
        https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/anova/how-to/two-way-anova/before-you-start/overview/
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
        '''
        Returns: mean square error for Two-way ANOVA.
        '''
        pass

    def between_group_var(self):
        '''
        Returns: between group variance for Two-way ANOVA.
        '''
        pass

    def within_group_var(self):
        '''
        Returns: within group variance for Two-way ANOVA.
        '''
        grand_mean = np.mean(self.data)

        pass

    def f_value(self):
        '''
        Returns: f-value for Two-way ANOVA.
        '''
        pass

    def p_value(self):
        '''
        Returns: p value for Two-way ANOVA.
        '''
        pass

    def residuals(self):
        '''
        Returns: sum of squares residuals for Two-way ANOVA.
        '''
        pass

    def r_square(self):
        '''
        Returns: r square for Two-way ANOVA.
        '''
        pass

    def adjusted_r_sq(self):
        '''
        Returns: adjusted r square for Two-way ANOVA.
        '''
        pass

    def print_summary(self):
        '''
        Prints summary statistics for Two-way ANOVA.
        '''
        sum_squares = self.sum_squares()
        mean_sq = self.mean_sq()
        mean_sq_err = self.mean_sq_err()
        between_group_var = self.between_group_var()
        within_group_var = self.within_group_var()
        f_value = self.f_value()
        p_val = self.p_val()
        r_sq = self.r_square()
        adj_r = self.adjusted_r_sq()
        cstr = "summary statistic"
        print(cstr.center(40, "="))
        # return print("mean: ", mean, "\nmedian: ", median, "\nmode: ", mode,
        #              "\nvar: ", var, "\nskewness: ", skewness, "\nkurtosis: ",
        #              kurtosis)

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
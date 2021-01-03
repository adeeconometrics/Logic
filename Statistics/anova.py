try:
    import numpy as np
    # import scipy as sci
    import scipy.special as ss
    import math as m
    # import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))


class BaseAnova:
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


class Anova1(BaseAnova):
    '''
    This class contains methods concerning One-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.
    '''
    def __init__(self, independent, dependent, adjust=True):
        super(Anova1, self).__init__(independent, dependent, adjust)

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
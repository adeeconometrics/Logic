try:
    import numpy as np
    import scipy as sci
    # import scipy.special as ss
    import math as m
    import matplotlib.pyplot as plt

except Exception as e:
    print("some modules are missing {}".format(e))


class Anova1:
    '''
    This class contains methods concerning One-way ANOVA which concerns itself to determine
    whether data from several groups(levels) have a common mean.

    Note that this class currently support equal sample sizes. 
    '''
    def __init__(self, data):
        self.data

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


class Anova2:
    pass


class Manova:
    pass


class Ancova:
    pass


class Mancova:
    pass
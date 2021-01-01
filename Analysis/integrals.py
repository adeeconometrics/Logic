import numpy as np
import math as m


class Univariate:
    '''
    This class contains numerical methods for approximating Univariate 
    Integrals.

    Args:

        func(function): function
        a(float): lower bound
        b(float | b>a): upper bound
        n(int):
    '''
    def __init__(self, func, a, b, n=10000):
        if a > b:
            raise Exception(
                'lower bound (a) should be less than upper bound(b).')
        self.func = func
        self.a = a
        self.b = b
        self.n = n

    def trapezoidal(self):
        '''
        this is an implementation of compositional trapazoidal method. 

        Returns: a tuple of approximation and error value
        '''
        a = self.a
        b = self.b
        h = float(b - a) / n
        result = 0.5 * self.func(a) + 0.5 * self.func(a)
        for i in range(1, self.n):
            result += self.func(a + i * h)
        result *= h
        return result

    def simpsons(self):
        pass

    def midpoint(self):
        '''
        this is an implementation of midpoint methods.

        Returns: a tuple of approximation and error value
        '''
        a = self.a
        b = self.b
        h = float(b - a) / n
        result = 0
        for i in range(self.n):
            result += self.func((a + h / 2.0) + i * h)
        result *= h
        return result

    def quadrature(self):
        pass

    pass


class Multivariate:
    pass

try:
    import numpy as np
except Exception as e:
    print("some modules are missing {}".format(e))

class groupedStatistics:
    # this does not account for already grouped data, there should be a function that formats the data structure
    """
    the groupedStatistics object contains methods for grouped statistics summary that describes
    the grouped dataset. 

    Args:
        data(list): raw dataset
        k(int): desired number of classes 
        initial(int): initial value of the lowest number. This is optional.
    """
    def __init__(self, data, k, initial=None):
        self.data = data
        self.k = k
        if initial is None:
            self.initial = 0
        self.initial = initial
        # derived value
        self.width = (max(data) -min(data))/k

    def class_limits(self):
        """
        Returns:
            list of limits of the class
        """
        k = self.k; init = self.initial
        for i in range(0,k):
            temp = init
            temp += k                
            limits = limits +[(init,init+k-1)]
            init = temp
        return limits

    def class_boundary(self):
        """
        Returns:
            list of tuples for the class boundaries
        """
        k = self.k; init = self.init
        for i in range(0,k):
            temp_a = init-0.5
            temp_b = temp_a+k
            boundary = boundary+[(temp_a, temp_b)] 
            temp_a = temp_b
            init = init+k
        return boundary

    def class_mark(self):
        """"
        It is defined as the average of the upper and lower class limits. 
        Returns:
            list of class marks 
        """"
        mark =  class_boundary(self)
        return [sum(i)/2 for i in mark]

    def class_frequency(self):
        """
        Returns:
            tuple of 
        """
        width = self.width; data = self.data
        size = max(width)+1 - min(width)
        size_set = set([min(width)+i for i in range(0, size)])
        return width, sum([data.count(i) for i in list(set(data)) if i in size_set])

    def cumulative_frequency(self):
        """
        Returns:
            list of cumulative frequency 
        """
        data = self.data
        return np.cumsum(data)
    
    def relative_frequency(self):
        """
        Returns:
            list of relative frequency
        """
        data = self.data
        sum_data  = sum(data)
        return [i/sum_data for i in data]

    def mean(self):
        """
        Returns:
            grouped mean
        """
        pass
    
    def median(self):
        """
        Returns:
            grouped median
        """
        pass
    
    def mode(self):
        """
        Returns:
            grouped mode
        """
        pass 
    
    def samp_variance(self):
        """
        Returns:
            grouped sample variance
        """
        pass
    
    def samp_std(self):
        """
        Returns:
            grouped sample stardard deviation
        """
        pass

    def fractiles(self):
        """"
        A fractile is the cut off point for a certain fraction of a sample. If your distribution is known, 
        then the fractile is just the cut-off point where the distribution reaches a certain probability
        """"
        pass
    
    def kurtosis(self):
        """
        Returns: 
            grouped kurtosis
        """
        pass

    def print_summary(self):
        pass 
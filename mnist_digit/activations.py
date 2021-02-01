import numpy as np

class Tanh:
    def __init__(self):
        pass

    def value(self, x):
        return np.tanh(x)

    def dvalue(self, x):
        return 1-self.value(x)**2

class ReLU:
    def __init__(self):
        pass 
    def value(self, x):
        return x * (x > 0)

    def dvalue(self, x):
        return 1. * (x > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha=alpha

    def value(self, x):
        y1 = ((x > 0) * x)                                                 
        y2 = ((x <= 0) * x * self.alpha)                                         
        return y1 + y2

    def dvalue(self, x):
        dx = np.ones_like(x).astype(np.float)
        dx[x < 0] = self.alpha
        return dx

class Softmax:
    def value(self, scores):
        return np.exp(scores)/sum(np.exp(scores))
    
    def dvalue(self, x):
        " only to be used with categorical cross entropy "
        return 1

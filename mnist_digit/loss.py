import numpy as np

class Cross_entropy:
    def __init__(self):
        pass

    def value(self, yp, y):
        return -1*np.sum(y.flatten()*np.log(1e-15 +yp.flatten()))

    def dvalue(self, yp, y):
        return yp-y
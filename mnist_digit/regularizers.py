
class Ridge:
    def __init__(self,n, lmda=0.1):
        self.lmda = lmda
        self.n = n

    def value(self, w1, w2):
        wsum = ((w1**2).ravel().sum())+((w2**2).ravel().sum())
        return (self.lmda/(2*self.n))*wsum

    def dvalue(self, w):
        return (self.lmda/self.n)*w

class Default:
    def __init__(self, n=1, lmda=0.1):
        self.lmda = lmda
        self.n = n

    def value(self, w1, w2):
        return 0
    def dvalue(self, w):
        return 0

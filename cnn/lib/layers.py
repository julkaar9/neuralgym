import numpy as np
import numpy.random as rnd

from .activations import Softmax # nopep8
from .loss import Cross_entropy # nopep8
from .regularizers import Default, Ridge # noqa
from .utils import norm, normalize

class Conv2D:
    def __init__(self, filters=1, filter_shape=3, padding='same'):

        self.no_kernals = filters
        self.filter_shape = filter_shape
        self.kernals = rnd.randn(filter_shape, filter_shape, filters)\
            / (self.filter_shape**2)
        self.padding = padding

        self.eta = 0.01
        self.sum_nw = 0

    def compile(self, input_shape):
        self.input_shape = input_shape
        if self.padding == 'same':
            h, w = input_shape
            h1, w1 = h + (self.filter_shape-1), w + (self.filter_shape-1)
            self.newimg = np.zeros((h1, w1))
        self.filter_map = np.zeros((h, w, self.no_kernals))

        output = self.forward(np.zeros(input_shape))
        self.output_shape = output.shape

    def gen_local_region(self, img):

        for r in range(self.input_shape[0]):
            for c in range(self.input_shape[1]):
                region = img[r:r+self.filter_shape, c:c+self.filter_shape]
                yield region, r, c

    def filter(self, img):
        h, w = img.shape
        ind1 = int(np.floor((self.filter_shape-1)/2))
        ind2 = int(np.ceil((self.filter_shape-1)/2))
       
        self.newimg.fill(img.min())
        self.newimg[ind1:h+ind2, ind1:w+ind2] = img

        for region, r, c in self.gen_local_region(self.newimg):
            self.filter_map[r, c, ] = np.sum(region[:, :, np.newaxis] *
                self.kernals, axis=(0, 1))

        return norm(self.filter_map)

    def forward(self, x):
        return self.filter(x)

    def backpropagation(self, dLdx):

        dLdw = np.zeros(self.kernals.shape)
        
        for region, r, c in self.gen_local_region(self.newimg):
            dLdw += dLdx[r, c]*region[:, :, np.newaxis]

        self.sum_nw += dLdw

    def update_gradients(self, mbs):
        self.kernals -= self.eta*self.sum_nw/mbs
        self.sum_nw = 0


class Maxpool:
    def __init__(self, pool_size=2, padding='same'):
        self.pool = pool_size
        self.padding = padding
        self.eta = 0.01

    def compile(self, input_shape):
        self.input_shape = input_shape
        p = self.pool
        if self.padding == 'same':
            h, w, no_kernals = input_shape
            h1, w1 = h + (h % p), w + (w % p)
            self.newimg = np.zeros((h1, w1, no_kernals))
        self.filter_map = np.zeros((h1//p, w1//p, no_kernals))

        output = self.forward(np.zeros(input_shape))
        self.output_shape = output.shape

    def gen_local_region(self, img):
        p = self.pool
        h, w, no_kernals = img.shape
        h1, w1 = h//p, w//p
        
        self.max_indices = np.zeros((h1*w1*no_kernals, 3))
        self.mi_len = 0

        for r in range(h1):
            for c in range(w1):
                region = img[r*p:r*p+p, c*p:c*p+p]

                argmx = np.argwhere(region == np.amax(region, axis=(0, 1)))
                argmx[:, 0] += (r*p)
                argmx[:, 1] += (c*p)
                argmx = argmx[np.unique(argmx[:, 2], axis=0, return_index=True)[1]]
                argmx = argmx[argmx[:, 2].argsort()]
                for i in range(len(argmx)):
                    self.max_indices[self.mi_len] = argmx[i]
                    self.mi_len += 1
                    
                yield region, r, c
        self.max_indices = self.max_indices[:self.mi_len]

    def filter(self, img):
        h, w, _ = img.shape
        p = self.pool
        self.newimg.fill(img.min())
        self.newimg[:h, :w, :] = img

        for region, r, c in self.gen_local_region(self.newimg):
            self.filter_map[r, c] = np.amax(region, axis=(0, 1))
        
        self.max_indices = self.max_indices.astype(np.int64)
        return self.filter_map

    def forward(self, x):
        return self.filter(x)

    def backpropagation(self, dLdx):
        p = self.pool
        self.reverse_max = np.zeros(self.newimg.shape)
        for (x, y, z) in self.max_indices:
            self.reverse_max[x, y, z] = dLdx[x//p, y//p, z]
    
        return self.reverse_max[:self.input_shape[0], :self.input_shape[1], :]


class Dense:
    def __init__(self, units, activation=Softmax()):
        self.units = units
        self.activation = activation
        self.eta = 0.01

        self.sum_nw, self.sum_nb = 0, 0

    def compile(self, input_shape):
        self.input_shape = input_shape
        self.init_weight(input_shape[0])
        output = self.forward(np.zeros(input_shape))
        self.output_shape = output.shape

    def init_weight(self, input_shape=1):
        self.w = rnd.normal(size=(self.units, input_shape))/input_shape
        self.b = rnd.normal(size=(self.units, 1))

    def forward(self, x):
        self.x = x
        self.z = np.matmul(self.w, x) + self.b
        self.a = self.activation.value(self.z)
        return self.a

    def backpropagation(self, dLda):
        dLdz = dLda*self.activation.dvalue(self.z)
        dLdw = np.matmul(dLdz, self.x.T)
        dLdb = dLdz
        dLdx = np.matmul(self.w.T, dLdz)

        self.sum_nw += dLdw
        self.sum_nb += dLdb
        return dLdx

    def update_gradients(self, mbs):

        self.w -= self.eta*self.sum_nw/mbs
        self.b -= self.eta*self.sum_nb/mbs
        self.sum_nw = self.sum_nb = 0


class Flatten:
    def __init__(self):
        pass

    def compile(self, input_shape):
        self.input_shape = input_shape
        output = self.forward(np.zeros(input_shape))
        self.output_shape = output.shape

    def forward(self, x):
        return x.flatten()[:, np.newaxis]

    def backpropagation(self, x):
        return x[:, 0].reshape(self.input_shape)

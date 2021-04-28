import numpy as np
import numpy.random as rnd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib.activations import Softmax # nopep8
from lib.loss import Cross_entropy # nopep8
from lib.regularizers import Default, Ridge # noqa
from lib.utils import * # noqa


np.set_printoptions(precision=5, suppress=True)
rnd.seed(2021)


def display(img):
    plt.imshow(img)
    plt.show()


class Conv2D:
    def __init__(self, filters=1, filter_shape=3, stride=1, padding='same'):

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

        #self.filter_map.fill(0)

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

        #self.filter_map = np.zeros((h1//p, w1//p, no_kernals))
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


class Model:
    def __init__(self):
        pass

    def sequential(self, layers):
        self.conv = layers[0]
        self.maxpool = layers[1]
        self.flatten = layers[2]
        self.dense = layers[3]
        return self

    def compile(self, input_shape, learning_rate=0.01, loss=Cross_entropy()):
        self.input_shape = input_shape
        self.conv.compile(input_shape=input_shape)
        self.maxpool.compile(input_shape=self.conv.output_shape)
        self.flatten.compile(input_shape=self.maxpool.output_shape)
        self.dense.compile(input_shape=self.flatten.output_shape)

        self.eta = learning_rate
        self.conv.eta = self.maxpool.eta = self.dense.eta = self.eta
        self.loss = loss

    def summary(self):
        print(self.conv.input_shape, self.conv.output_shape)
        print(self.maxpool.input_shape, self.maxpool.output_shape)
        print(self.flatten.input_shape, self.flatten.output_shape)
        print(self.dense.input_shape, self.dense.output_shape)

    def get_loss(self, yp, y):

        return self.loss.value(yp, y)

    def feedforward(self, x):
        self.c1 = self.conv.forward(x)
        self.m1 = self.maxpool.forward(self.c1)
        self.f1 = self.flatten.forward(self.m1)
        self.a1 = self.dense.forward(self.f1)
        return self.a1

    def backpropagation(self, x, y):
        dLda1 = self.loss.dvalue(self.a1, y)
        dLdf1 = self.dense.backpropagation(dLda1)
        dLdm1 = self.flatten.backpropagation(dLdf1)
        dLdc1 = self.maxpool.backpropagation(dLdm1)
        dLdx = self.conv.backpropagation(dLdc1)

    def update_gradients(self):
        self.dense.update_gradients(self.mbs)
        self.conv.update_gradients(self.mbs)

    def fit(self, X, Y, val_set=None, mbs=32, epoch=1):
        ce = 0
        self.mbs = mbs
        print(X.shape, Y.shape)
        for i in range(epoch):
            ce, yt = 0, 0
            for ind, (x, y) in enumerate(zip(X, Y)):
                a2 = self.feedforward(x)
                
                if np.argmax(a2) == np.argmax(y):
                    yt += 1
                ce += self.get_loss(a2, y)
                self.backpropagation(x, y)
                if (ind+1) % self.mbs == 0:
                    self.update_gradients()
            self.update_gradients()
            print('ce loss', ce)
            print('train_acc', yt*100/X.shape[0], end='')
            print(' val_acc', accuracy(self, val_set[0], val_set[1]))

    def predict(self, x):
        return np.argmax(self.feedforward(x))

    def save(self, name):
        np.savez(name+'.npz',
                convker=self.conv.kernals,
                densew=self.dense.w,
                denseb=self.dense.b)

    def load(self, name):
        wgt = np.load(name+'.npz')
        self.conv.kernals = wgt['convker']
        self.dense.w = wgt['densew']
        self.dense.b = wgt['denseb']


def main():

    X_train, y_train, X_val, y_val = load_mnist(reshape=True, n_rows=1000)
    model = Model().sequential([Conv2D(filters=8, filter_shape=3),
                                Maxpool(),
                                Flatten(),
                                Dense(10)])
    model.compile(X_train[0].shape)

    model.summary()
    model.load('cnn5')
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    # display(X_val[0])
    # y1 = model.conv.forward(X_val[0])
    # f = 1
    for f in [0,1,2,3,4,5,6,7]:
         #display(y1[:, :, f])
         #display(model.maxpool.forward(y1)[:, :, f])

        display(model.conv.kernals[:, :, f])
    # display(X_train[0])
    #model.fit(X_train, y_train, (X_val, y_val), mbs=32, epoch=10)
    #model.save('cnn5')
    #annote_test(model, X_val[:9], 3, 3)


main()
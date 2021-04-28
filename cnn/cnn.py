import numpy as np
import numpy.random as rnd
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

import cv2

from lib.activations import Softmax
from lib.loss import Cross_entropy
from lib.regularizers import Default, Ridge
from lib.utils import onehotcode, load_mnist, norm, normalize


np.set_printoptions(suppress=True)
rnd.seed(2021)


def display(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Conv2D:
    def __init__(
            self, filters=1, filter_shape=3, stride=1, padding='same',
            input_shape=None):
        # self.kernals = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        self.no_kernals = filters
        self.filter_shape = filter_shape
        self.kernals = rnd.randn(filters, filter_shape, filter_shape)
        self.padding = padding
        self.eta = 0.01

        if input_shape:
            self.input_shape = input_shape
            self.feedforward(np.zeros(input_shape))

    def gen_local_region(self, img):
        h, w = img.shape

        for r in range(h-2):
            for c in range(w-2):
                region = img[r:r+3, c:c+3]
                yield region, r, c

    def filter(self, img):
        if self.padding == 'same':
            h, w = img.shape
            h1, w1 = h+2, w+2
            newimg = np.zeros((h1, w1))
            newimg[1:h+1, 1:w+1] = img
        self.filter_map = np.zeros((h, w, self.no_kernals))

        for region, r, c in self.gen_local_region(img):
            self.filter_map[r, c] = np.sum(region*self.kernals,  axis=(1, 2))

        return norm(self.filter_map)

    def forward(self, x):
        z = self.filter(x)
        self.output_shape = z.shape
        return z


class Maxpool:
    def __init__(self, pool_size=2, padding='same', input_shape=None):
        self.pool = pool_size
        self.padding = padding
        self.eta = 0.01

        if input_shape:
            self.input_shape = input_shape
            self.feedforward(np.zeros(input_shape))

    def gen_local_region(self, img):
        p = self.pool
        h, w, no_kernals = img.shape
        h1, w1 = h//p, w//p
        for r in range(h1):
            for c in range(w1):
                region = img[r*p:r*p+p, c*p:c*p+p]
                yield region, r, c

    def filter(self, img):
        p = self.pool
        if self.padding == 'same':
            h, w, no_kernals = img.shape
            h1, w1 = h + (h % p), w + (w % p)
            newimg = np.zeros((h1, w1, no_kernals))
            newimg[:h, :w, :] = img

        self.filter_map = np.zeros((h1//p, w1//p, no_kernals))
        for region, r, c in self.gen_local_region(img):
            self.filter_map[r, c] = np.amax(region, axis=(0, 1))

        return self.filter_map

    def forward(self, x):
        z = self.filter(x)
        self.output_shape = z.shape
        return z


class Dense:
    def __init__(self, units, activation=Softmax(), input_shape=1):
        self.units = units
        self.activation = activation
        self.eta = 0.01

        self.sum_nw, self.sum_nb = 0, 0

        if input_shape:
            self.input_shape = input_shape
            self.feedforward(np.zeros(self.input_shape))

    def init_weight(self, input_shape=1):
        self.w = rnd.normal(size=(self.units, input_shape))/input_shape
        self.b = rnd.normal(size=(self.units, 1))

    def forward(self, x):
        # print('dense\n', self.w.shape, x.shape, self.b.shape)
        self.z = np.matmul(self.w, x) + self.b
        # print(self.z.shape)
        self.a = self.activation.value(self.z)
        # print(' ', self.a.shape)
        self.output_shape = self.a.shape
        return self.a

    def backpropagation(self, dLda, xf):
        # print('back')
        # print(dLda.shape)
        dLdz = dLda*self.activation.dvalue(self.z)
        # print(dLdz.shape)
        dLdw = np.matmul(dLdz, xf.T)
        # print(dLdw.shape)
        dLdb = dLdz
        # print(dLdb.shape)
        dLdx = np.matmul(self.w.T, dLdz)
        # print(dLdx.shape)

        self.sum_nw += dLdw
        self.sum_nb += dLdb
        return dLdx

    def update_gradients(self, mbs):

        self.w -= self.eta*self.sum_nw/mbs
        self.b -= self.eta*self.sum_nb/mbs
        self.sum_nw = self.sum_nb = 0


class Flatten:
    def __init__(self, input_shape=None):
        if input_shape:
            self.input_shape = input_shape

    def feedforward(self, x):
        output = x.flatten()[:, np.newaxis]
        self.output_shape = output

        return output

    def backpropagation(self, x):
        return x.reshape(self.input_shape)


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
        self.conv.__init__(input_shape=input_shape)
        self.maxpool.__init__(input_shape=self.conv.output_shape)
        self.flatten.__init__(input_shape=self.maxpool.output_shape)
        self.dense.__init__(input_shape=self.flatten.output_shape)
        self.dense.init_weight(self.maxpool.output_shape)

        self.eta = learning_rate
        self.conv.eta = self.maxpool.eta = self.dense.eta = self.eta
        self.loss = loss

    def get_loss(self, yp, y):

        return self.loss.value(yp, y)

    def feedforward(self, x):
        # print(x.shape)
        self.c1 = self.conv.forward(x)
        # print(self.c1.shape)
        self.m1 = x  # self.maxpool.forward(self.c1)
        # print(self.m1.shape)
        self.f1 = self.m1.flatten()[:, np.newaxis]
        # print(self.f1.shape)
        self.a1 = self.dense.forward(self.f1)
        # print(self.a1.shape)
        return self.a1

    def backpropagation(self, x, y):
        # print('back')
        dLda1 = self.loss.dvalue(self.a1, y)
        # print(dLda1.shape)
        dLdf1 = self.dense.backpropagation(dLda1, self.f1)
        # print(dLdf1.shape)
        dLdm1 = dLdf1[:, 0].reshape(self.m1.shape)

    def update_gradients(self):
        self.dense.update_gradients(self.mbs)

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
            print('train_acc', yt*100/X.shape[0])  # ,'val_acc', self.accuracy\
            # (val_set[0],val_set[1]))


img = np.array(cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE))
# print(img.shape)
# img = Conv2D(filters=2).filter(norm(img))
# print(img.shape)
# #display(img[:,:,1])
# img = Maxpool().filter(norm(img))
# print(img.shape)
# #display(img[:,:,1])
# y = Dense(units=10, input_shape=(113*113*2)).forward(img.flatten().reshape(25538, 1))
# print(y.shape)

X_train, y_train, X_val, y_val = load_mnist(reshape=True, n_rows=1000)
model = Model().sequential([Conv2D(filters=2),
                            Maxpool(),
                            Flatten(),
                            Dense(10)])
model.compile(X_train[0])
X_train = normalize(X_train)

# display(X_train[0])
model.fit(X_train, y_train, mbs=1, epoch=10)

import numpy as np
from .loss import Cross_entropy
from .utils import accuracy

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
            if val_set is not None:
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

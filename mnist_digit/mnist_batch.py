
import numpy as np
import numpy.random as rnd

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import seaborn as sns

from activations import Tanh, ReLU, LeakyReLU, Softmax
from loss import Cross_entropy
from regularizers import Default, Ridge
from utils import onehotcode

rnd.seed(1973)
np.set_printoptions(suppress=True)


class NN:
    def __init__(self):
        self.eta = 0.05
        self.mbs = 32

        self.sum_nw2, self.sum_nb2, self.sum_nw1, self.sum_nb1 = 0, 0, 0, 0

    def sequential(self, network=[1,1,1], activation=[ReLU(), Softmax()], 
                    loss=Cross_entropy(), regu=Default(), weight_type='default'):

        self.net = network
        self.activation1 = activation[0]
        self.activation2 = activation[1]
        self.loss = loss

        self.regu = regu
        self.init_weight(weight_type)


    def init_weight(self, type='default'):
        net = self.net
        self.w1 = self.get_weight(size=(net[1], net[0]),type=type)
        self.b1 = self.get_weight(size=(net[1], 1))

        self.w2 = self.get_weight(size=(net[2], net[1]),type=type)
        self.b2 = self.get_weight(size=(net[2], 1))

        print(self.w1.shape, self.b1.shape)
        print(self.w2.shape, self.b2.shape)

    def get_weight(self, size=(1,1), type='default'):
        if type=='default':
            return rnd.normal(size=size)
        elif type=='glorot_normal':
            return rnd.normal(size=size)/size[1]
        elif type=='he_normal':
            return rnd.normal(size=size)*(2/size[1])
        elif type=='abs':
            return np.abs(rnd.normal(size=size))/size[1]

    def accuracy(self,X,Y):
        acc = 0
        for x, y in zip(X,Y):
            if self.predict(x)==y:
                acc += 1
         
        return 100*acc/Y.shape[0]

    def predict(self, x):
        return np.argmax(self.feedforward(x))

    def save_model(self, name=""):
        np.save(name+'w1.npy', self.w1)
        np.save(name+'b1.npy', self.b1)
        np.save(name+'w2.npy', self.w2)
        np.save(name+'b2.npy', self.b2)

    def load_model(self, name=""):
        self.w1 = np.load(name+'w1.npy')
        self.b1 = np.load(name+'b1.npy')
        self.w2 = np.load(name+'w2.npy')
        self.b2 = np.load(name+'b2.npy')

    def get_loss(self,yp,y):

        return self.loss.value(yp, y) +self.regu.value(self.w1, self.w2)


    def feedforward(self,x):

        self.z1 = np.matmul(self.w1,x)+self.b1
        self.a1 = self.activation1.value(self.z1)

        self.z2 = np.matmul(self.w2,self.a1)+self.b2
        self.a2 = self.activation2.value(self.z2)

        return self.a2
        

    def backpropagation(self,x,y):
        dLdz2 = self.loss.dvalue(self.a2, y)*self.activation2.dvalue(self.z2)
         
        dLdw2 = np.matmul(dLdz2,self.a1.T)
       
        dLdb2 = dLdz2*1
       
        dLdz1 = np.matmul(self.w2.T,dLdz2)*self.activation1.dvalue(self.z1)
        dLdw1 = np.matmul(dLdz1,x.T)

        dLdb1 = dLdz1*1

         
        self.sum_nw2 += self.eta*dLdw2 
        self.sum_nb2 += self.eta*dLdb2

        self.sum_nw1 += self.eta*dLdw1
        self.sum_nb1 += self.eta*dLdb1

    def update_gradients(self):

        self.w2 -= (self.sum_nw2/self.mbs + self.eta*self.regu.dvalue(self.w2))
        self.b2 -= self.sum_nb2/self.mbs

        self.w1 -= (self.sum_nw1/self.mbs + self.eta*self.regu.dvalue(self.w1))
        self.b1 -= self.sum_nb1/self.mbs

        self.sum_nw2, self.sum_nb2, self.sum_nw1, self.sum_nb1 = 0, 0, 0, 0


    def fit(self, X ,Y, X_test, y_test, mbs, epoch=10):
        ce = 0
        self.mbs = mbs

        for i in range(epoch):
            ce, yt = 0, 0
            for ind, (x, y) in enumerate(zip(X,Y)):
                a2 = self.feedforward(x)
                if np.argmax(a2)==np.argmax(y):
                    yt += 1
                ce += self.get_loss(a2, y)
            
                self.backpropagation(x,y)

                if ind%self.mbs==0:
                    self.update_gradients()
            
            self.update_gradients()
            print('ce loss', ce)
            print('train_acc',yt*100/X.shape[0],'val_acc', self.accuracy(X_test,y_test))
            

    def annote_test(self,X,r,c):
        fig, axes = plt.subplots(r,c, figsize=(10,10))
        for i,ax in enumerate(axes.reshape(-1)):
            ax.set_axis_off()
            ax.imshow(X[i])
            yp = self.predict(X[i].reshape((784,1)))
           
            ax.text(0.5, 1.5,yp, color='black', \
                bbox=dict(facecolor='white', edgecolor='black'))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout() 
        plt.show()
    
    def weight_heatmap(self):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,20))
        sns.heatmap(self.w1, ax=ax1,  cmap='icefire')
        sns.heatmap(self.w2, annot=True, ax=ax2, cmap='viridis')
        plt.show()
        
df = pd.read_csv(p+'mnist_train.csv')

train = df.sample(frac=0.8,random_state=200)
val = df.drop(train.index)

yr = train.iloc[:,0].to_numpy()
X_train, y_train = train.iloc[:,1:].to_numpy()/255.0, onehotcode(yr)

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],y_train.shape[1],1))
print(X_train.shape, y_train.shape)


X_val, y_val = val.iloc[:,1:].to_numpy()/255.0, val.iloc[:,0].to_numpy()

X_val = X_val.reshape((X_val.shape[0],X_val.shape[1],1))
print(X_val.shape, y_val.shape)


nn = NN()
nn.sequential(network=[784, 128, 10],
            activation=[Tanh(), Softmax()],
            loss=Cross_entropy(),
            regu=Ridge(n=X_train.shape[0], lmda=5),
            weight_type='glorot_normal')
nn.load_model('tanL2128')
#nn.fit(X_train,y_train,X_val,y_val,32,2)
#nn.save_model('tanL2128')
nn.weight_heatmap()

df2 = pd.read_csv('mnist_test.csv')
X_test, y_test = df2.iloc[:,1:].to_numpy()/255.0,df2.iloc[:,0].to_numpy()
X_test = X_test.reshape((X_test.shape[0],28,28))
print(X_test.shape, y_test.shape)
nn.annote_test(X_test[:100],10,10)


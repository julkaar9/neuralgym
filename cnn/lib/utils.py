import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display(img):
    plt.imshow(img)
    plt.show()

def onehotcode(a, mx=None):
    if mx == None:
        mx = a.max()+1
    b = np.zeros((a.size, mx))
    b[np.arange(a.size), a] = 1
    return b


def norm(x, a=-1, b=1):
    if (x.max()-x.min()) != 0:
        return (b-a)*((x-x.min())/(x.max()-x.min()))+a
    else:
        return x


def normalize(X, a=-1, b=1):
    for i in range(len(X)):
        X[i] = norm(X[i], a, b) 
    return X


def accuracy(model, X, Y):
    acc = 0
    for x, y in zip(X,Y):
        if model.predict(x)==y:
            acc += 1  
    return 100*acc/Y.shape[0]
    

def annote_test(model, X, r, c):
    fig, axes = plt.subplots(r, c, figsize=(10, 10))
    for i, ax in enumerate(axes.reshape(-1)):
        ax.set_axis_off()
        ax.imshow(X[i])
        yp = model.predict(X[i])
        
        ax.text(0.5, 1.5, yp, color='black',
                bbox=dict(facecolor='white', edgecolor='black'))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.show()

def load_mnist(reshape=False, n_rows=None):
    df = pd.read_csv('mnist_train.csv', nrows=n_rows)

    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)

    yr = train.iloc[:,0].to_numpy()
    X_train, y_train = train.iloc[:, 1:].to_numpy().astype(np.float), onehotcode(yr, 10)
 
    y_train = y_train.reshape((y_train.shape[0],y_train.shape[1], 1))
    
    X_val, y_val = val.iloc[:, 1:].to_numpy().astype(np.float), val.iloc[:, 0].to_numpy()

    X_val = X_val.reshape((X_val.shape[0],X_val.shape[1], 1))
    
    if reshape:
        X_train = X_train.reshape((X_train.shape[0], 28, 28))
        X_val = X_val.reshape((X_val.shape[0], 28, 28))
    else:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    return X_train, y_train, X_val, y_val

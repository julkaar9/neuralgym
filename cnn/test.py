import numpy as np 
np.random.seed(2021)

x = np.random.randint(0, 4, (2, 2, 2))
y = np.random.randint(0, 4, (2, 2))
print(x[:, :, 0])
print(x[:, :, 1])
print(y)
z = x*y[:,:,np.newaxis]
print(z.shape)
print(z[:, :, 0])
print(z[:, :, 1])

x = np.random.randint(0, 6, (3, 3)).astype(np.float64)
print(x)
c = Conv2D()
#c.compile(x.shape)
y1 = c.forward(x)
print(y1[:,:,0])
m = Maxpool()
m.compile(y1.shape)
y2 = m.forward(y1)

print(y2[:, :, 0],'\n')

y3 = m.backpropagation(y2)

print(y3[:, :, 0])

y4 = c.backpropagation(y3)
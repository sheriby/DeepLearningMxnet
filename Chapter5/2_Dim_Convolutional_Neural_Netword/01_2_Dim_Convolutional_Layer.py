#!/usr/bin/env python
# coding: utf-8

# # 2-Dimension Convolutional Layer
# `Convolutional Neural network` is neural netword with `Convolution Layer`, which is usually applied to **processing images data**.

# ## 2-Dimension Cross-correlation Operation
# In convolutional neural network, the output of corss-correlation operation between 2-dim input array and 2-dim kernel array is 2-dimension array.

# For example, the input array is 3*3.  
# $$\begin{bmatrix}0 & 1 & 2 \\ 3 & 4 & 5 \\ 6 & 7 & 8\end{bmatrix}$$
# <br>And the kernel array is 2*2.
# $$\begin{bmatrix}0 & 1 \\ 2 & 3\end{bmatrix}$$
# <br>Through corss-correlation operation, we get a 2-dimension as following.
# $$\begin{bmatrix}19 & 23\\ 37 & 43\end{bmatrix}$$

# In 2-dimension cross-correlation operation, the order of operation is from top to bottom, from left to right.
# Therefore, 19 is the result of these two array.
# $$\begin{bmatrix}0 & 1 \\ 3 & 4\end{bmatrix}\quad\begin{bmatrix}0 & 1 \\ 2 & 3\end{bmatrix}$$
# <br>
# $$0 \times 0 + 1 \times 1 + 3 \times 2 + 4 \times 3 = 19$$
# 23, 37, 43 is the same operation.

# In[1]:


from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):
    """
        @param X the input array
        @param K the kernel array
        @return the result of cross-correlation operation
    """
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # the output array
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# In[7]:


X = nd.arange(9).reshape(3, 3)
Y = nd.arange(4).reshape(2, 2)
X, Y


# In[6]:


corr2d(X, Y)


# ## 2-Dimension Convolutional Layer
# Using function corr2d to bulid a convolutional layer.

# In[8]:


class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))
        
    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


# ## Edge Detection Of A Object In a Image
# The simplest application of convolutional layer is **edge detection of a object in a image**, which is finding where pixel changes.  
# Assume here is a image with a height of 6 pixel and a width of 8 pixel.(6*8 array where 0 means `black` and 1 means `white`.)

# In[9]:


X = nd.ones(shape=(6, 8))
X[:, 2:6] = 0
X


# Then constructing a convolutional kernel with a height of 1 pixel and a width of 2 pixel. When cross-correlation operating with input array, if the horizontal elements is are the same, output 0 otherwise non-0. 

# In[12]:


K = nd.array([[1, -1]])

Y = corr2d(X, K)
Y


# It is obvious to see that the value of object in edge is 1 or -1, and the value other part is 0.
# From this we can see that convolution layer can **effectivly represent local area by repeatly using convolutional kernel**.

# ## Study Kernel Array By Data
# You know in the above example, we use the kernel array [[1, -1]]. But why this? And can other array ok?
# For instance, we use kernel array[[1, 2]].

# In[15]:


K = nd.array([[1, 2]])

Y = corr2d(X, K)
Y


# It is bad! So suitable kernel array is necessary!  
# Here we will use data to train the kernel array, just like training weight and bias parameters before.  
# When it comes to training model, loss function and optimization algorithm are must. Here we will use square loss function and gradient descent algorithm just like before.

# In[16]:


# construct 2d convolutional layer with 1 input channel and 1*2 kernel array.
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 4d array => whose format is (sample, number of input channel, weight, height).
X = X.reshape(shape=(1, 1, 6, 8))
Y = Y.reshape(shape=(1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # For simplicity, here ignoring the bias.
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad() # gradient descent
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f'%(i+1, l.sum().asscalar()))


# In[17]:


conv2d.weight.data().reshape((1, 2))
# it is very close to [1, -1]


# ## Corss-correlation Operation And Convolutional Operation
# As a metter of fact, corss-correlation operation is similiar to convolutional operation. Flip the kernel array left to right and up to down, and then get a new kernel array. Use new kernel array to do corss-correlation operation with input array, which is the result of convolutional operation. So it is obvious to see that in meachine learning, these two operation is equivalent.

# ## Feature Map And Receptive Field.
# The output 2darray of 2d convolutional layer can be seen as the `Feature Map` which is the representation of the input in some level.  
# `Receptive Field` is the area which possibly influence the forward pass calculation of element x.

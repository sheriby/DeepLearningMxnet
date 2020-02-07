#!/usr/bin/env python
# coding: utf-8

# # Pooling Layer
# When talking about finding the edge of object before, we construct a convolution kernel to find the position where pixel changes.  
# While in true image, the target object will not be always at same position. It may be a little offset, which will cause that the output of same edge is found in different position, influenting the pattern recognition later.  
# Pooling layer is to **relieve the over sensitivity of the convolution layer**.

# ## 2D Maximum Pooling Layer And Average Pooling Layer
# Similiar to convolutional layer, pooling layer calculate output of a pooling window for input array. Different from corss-correlation operation in convolutional layer, pooling layer **get the maximum value from pooling window as result**, which is also called as `maximum pooling` or `average pooling`.

# For example, through $2 \times 2$ pooling layer.
# $$\begin{bmatrix}0&1&2\\3&4&5\\6&7&8\end{bmatrix}$$
# becomes
# $$\begin{bmatrix}4&5\\7&8\end{bmatrix}$$
# because
# $$\begin{cases}
# \max(0, 1, 3, 4) = 4\\
# \max(1, 2, 4, 5) = 5\\
# \max(3, 4, 6, 7) = 7\\
# \max(4, 5, 7, 8) = 8
# \end{cases}$$

# Using $2 \times 2$ pooling layer, if the recognition pattern moves 1 element, we can still detect it.

# In[1]:


from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


# In[2]:


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
pool2d(X, (2, 2))


# In[3]:


pool2d(X, (2, 2), 'avg')


# ## Padding And Stride
# Similiar to convolutional layer, we can also adjust padding and stride in pooling layer to change the shape of ouput array.

# In[14]:


X = nd.arange(16).reshape((1, 1, 4, 4))
X


# In[9]:


pool2d = nn.MaxPool2D(3) # 3*3 pooling layer
# defaultly, stride in MaxPool2D is equal to the shape of pooling window 
# so here stride equals 3
pool2d(X)


# In[10]:


pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)


# In[11]:


pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)


# ## Multiply Channel

# In[15]:


X = nd.concat(X, X+1, dim=1)
X


# In[16]:


pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)


# The number of channel does not change.

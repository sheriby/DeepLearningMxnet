#!/usr/bin/env python
# coding: utf-8

# # Multi-Input And Multi-Ouput Channel
# Before input array and output array used are all 2d array, but there may be more dimension in true data. For example, colorful image has three other color channel RGB apart from height and width. Therefore, it can be represented as 3d array(3 * h * w) and we call 3 here as `channel`.

# ## Multi-Input Channel
# When there are multi-input channel, we shall construct a same multiply kernel channel to do corss-correlation operation. 

# The following is the example about the corss-correlation operation of multi-input channel.  
# Here is a input array with two channel.
# $$\left[\begin{bmatrix}0&1&2\\3&4&5\\6&7&8\end{bmatrix},\quad
# \begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix}\right]$$
# Here is a kernel array with two channel as well.
# $$\left[\begin{bmatrix}0&1\\2&3\end{bmatrix},\quad \begin{bmatrix}1&2\\3&4\end{bmatrix}\right]$$
# In each channel, do corss-correlation with 2d input array and 2d kernel array and then add the result by channel. That is the result.
# $$\begin{bmatrix}0&1&2\\3&4&5\\6&7&8\end{bmatrix} * \begin{bmatrix}0&1\\2&3\end{bmatrix}$$
# $$+$$
# $$\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix} * \begin{bmatrix}1&2\\3&4\end{bmatrix}$$
# $$=$$
# $$\begin{bmatrix}56&72\\104&120\end{bmatrix}$$

# In[1]:


import d2lzh as d2l
from mxnet import nd

def corr2d_multi_in(X, K):
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])


# In[5]:


X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
X


# In[6]:


K = nd.array([[[0, 1], [2, 3]], [[1, 2],[3, 4]]])
K


# In[7]:


corr2d_multi_in(X, K)


# ## Multi-Output Channel
# When there are multiply input channel, the output channel is always one due to accumulating all channel. To get multi-ouput channel, we can create $c_i \times k_h \times k_w$ for each channel and concatenate them. Therefore the shape of kernel array is $c_o \times c_i \times k_h \times k_w$

# In[8]:


def corr2d_multi_in_out(X, K):
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])


# In[9]:


K = nd.stack(K, K+1, K+2)
K, K.shape


# In[10]:


corr2d_multi_in_out(X, K)


# ## $1 \times 1$ Convolutional Layer
# If $k_h = k_w = 1$, usually we call it as $1 \times 1$ convolutional layer and coovolutional operation here is called as $1 \times 1$ convolution.  
# $1\times 1$ is the smallest window size, so it loss the featrue of it's approaching element.  
# **$1 \times 1$ convolutional layer is equivalent to fully connectional layer**.

# In[13]:


def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w)) # data sample
    K = K.reshape((c_o, c_i)) # feature 
    Y = nd.dot(K, X) # matrix dot in fully connectional layer
    return Y.reshape((c_o, h, w))


# In[14]:


X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().asscalar() < 1e-6


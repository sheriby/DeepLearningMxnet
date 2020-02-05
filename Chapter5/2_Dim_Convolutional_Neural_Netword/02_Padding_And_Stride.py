#!/usr/bin/env python
# coding: utf-8

# # Padding And Stride
# Generally speaking, if the shape of input array is $n_h \times n_w$ and the shape of convolutional kernel is $k_h \times k_w$, then the shape of output will be 
# $$(n_h - k_h + 1) \times (n_w - k_w + 1)$$
# Therefore, it is obvious that the shape of output is based on the shape of input array and convolutional kernel. And Here we can use two hyperparameters--`Padding` and `Stride` to change the shape of ouput.

# ## Padding
# Padding means that **Fill element in both sizes of height and width of the input array.**(Usually the element is **0**). Using this method, we can change the shape of the input array and thus change the shape of the output array.
# Generally speaking, if we fill **$p_h$** row in the height and **$p_w$** column in the width. Then the shape of the output will be
# $$(n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)$$

# In most of conditions, we will set $p_h = k_h - 1$ and $p_w = k_w - 1$ to make the output array has same height and width. If $p_h$ is odd number, we will padding $p_h/2$ row in both size of height. If $p_h$ is even number, we will padding $\lceil\ p_h/2\ \rceil$ row in top size and pdding $\lfloor\ p_h/2\ \rfloor$. $p_w$ is similiar to.

# In convolutional neural network, we usually use convolutional kernel with odd height and odd weight in order to fill same row or column in both size of input array.

# In[1]:


from mxnet import nd
from mxnet.gluon import nn

def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) is (batch size, input channel)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # we don't care batch and channel.


# In[3]:


conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
res = comp_conv2d(conv2d, X)
res, res.shape
# 8 - 3 + 2*1 + 1 = 8


# In[4]:


conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
res = comp_conv2d(conv2d, X)
res, res.shape
# 8 - 5 + 2*2 + 1 = 8
# 8 - 3 + 2*2 + 1 = 8


# ## Stride
# When doing 2D corss-correlation operation, the convolutional window move from left to right and from top to bottom. The number of row or column that we move each time is `Stride`.
# At present, each time we move one step. Therefore, the stride is 1. We can use bigger stride, and then the output array will be smaller.

# Generally speaking, when the stride in height and weight are $s_h$ and $s_w$, the shape of output array as follows.
# $$\left\lfloor(n_h - k_h + p_h + s_h)\ /\ s_h\right\rfloor \times \left\lfloor(n_w - k_w + p_w + s_w)\ /\ s_w\right\rfloor$$
# Set $p_h = k_h - 1$, we can get 
# $$\lfloor(n_h + s_h -1)\ /\ s_h\rfloor \times \lfloor(n_w + s_w - 1)\ /\ s_w\rfloor$$
# And if $n_h mod s_h = 0$, we can get
# $$(n_h\ /\ s_h) \times (n_w\ /\ s_w)$$

# In[5]:


conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
res = comp_conv2d(conv2d, X)
res, res.shape
# floor[(8 - 3 + 2*1 + 2)/2] = floor(9/2) = 4


# In[6]:


conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
# floor[(8 - 3 + 2*0 + 3)/3] = floor(8/3) = 2
# floor[(8 - 5 + 2*1 + 4)/4] = florr(9/4) = 2


# For brevity, we say the padding is $(p_h, p_w)$ and the stride is $(s_h, s_w)$. If $p_h = p_w = p$ or $(s_h = s_w = s)$, we say that padding is $p$ or stride is $s$.

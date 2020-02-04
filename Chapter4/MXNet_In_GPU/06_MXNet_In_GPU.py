#!/usr/bin/env python
# coding: utf-8

# # MXNet In GPU
# Above all, there is at least a GPU in your computer, and then you can use MXNet in GPU. There is none `Graphics Card` in most of MacBook, so it is impossible to do GPU calculation. However, you can do it in cluod computer by using `ssh`.
# You must download `NVIDIA cuda` if you have a `NVIDIA Graphics Card`, and then download corresponding version of `MXNet`(**Uninstall the MXNet version without GPU first**).
# For example, I have a Graphics Card NVIDIA MX150. Therefore I download `cuda10.0` and `mxnet-cuda100`. 

# In[2]:


import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

mx.cpu(), mx.gpu() # cup(0) and gpu(0)
# if having more cpu and gpu, you can also use `cpu(1)`, `gpu(1)`, `gpu(2)` and so on.


# In[4]:


x = nd.array([1, 2, 3])
x # ndarray is in cpu(0) defaultly!


# In[5]:


x.context # use `context` to get which device it in.


# ## Store In GPU

# In[6]:


a = nd.array([1, 2, 3], ctx=mx.gpu()) 
# use `ctx`, which is the abbreviation of `context`, to specify storage device 
a


# In[7]:


y = x.copyto(mx.gpu()) # copy data from cpu to gpu
# This operation is time-consuming. However, in many cases, we have to copy data from cpu to gpu.
# such as printing ndarray or transfering ndarray to numpy and so on.


# In[9]:


z = x.as_in_context(mx.gpu()) # this function will also copy data from cpu to gpu
z


# In[10]:


# but if the source data and destination data is in same context.
# these two variable will share data.
y.as_in_context(mx.gpu()) is y # it is true


# In[11]:


y.copyto(mx.gpu()) is y # it is false


# ## Calculation In GPU

# In[12]:


(z + 2).exp() * y
# z and y is in gpu(0), so the result is in gpu(0)


# In[15]:


# x + y 
# x is in cpu(0), while y is in gpu(0).
# we cannot calculation these two variable unless we make them in the same device


# ## GPU Calculation With Gluon

# In[16]:


net = nn.Sequential()
net.add(nn.Dense(1))

net.initialize(ctx=mx.gpu()) # specify using gpu


# In[17]:


net(y) # result is in gpu(0) [y is must in gpu(0) as well]


# In[19]:


net[0].weight.data() # parameter is also in gpu(0)


# In[29]:


x = nd.arange(1, 250000001).reshape(5000, 5000)
x


# In[41]:


from time import time
start = time()
for i in range(10000):
    nd.dot(x+i, x+i+10)
time() - start


# In[42]:


y = x.copyto(mx.gpu())


# In[43]:


start = time()
for i in range(10000):
    nd.dot(y+i, y+i+10000)    
time() - start


# It seems that GPU is a little faster....but not obvious.  
# Prohapsly, my GPU is extreamly bad, as my graphics card is low!!!

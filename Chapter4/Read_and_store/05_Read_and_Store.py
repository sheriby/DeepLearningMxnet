#!/usr/bin/env python
# coding: utf-8

# # Read and Store

# ## Read and Write NDArray

# In[1]:


from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)
nd.save('x', x) # store data of x into a file named 'x' similarly.


# In[2]:


x2 = nd.load('x')
x2


# In[3]:


y = nd.zeros(4)
nd.save('xy', [x, y]) # store a list of ndarray
x2, y2 = nd.load('xy') # read data
x2, y2


# In[5]:


mydict = {'x' : x, 'y' : y}
nd.save('mydict', mydict) # we can even store a dictory
mydict2 = nd.load('mydict')
mydict2


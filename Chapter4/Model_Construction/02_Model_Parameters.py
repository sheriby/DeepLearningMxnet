#!/usr/bin/env python
# coding: utf-8

# # Model Parameters
# The accession, initialization and sharing of the model parameters.

# In[2]:


from mxnet import init, nd
from mxnet.gluon import nn 

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize() # use default initialization method.

X = nd.random.uniform(shape=(2, 20))
Y = net(X)


# In[3]:


X, Y


# # Access Model Parameters

# In[4]:


net[0].params, type(net[0].params) # net[0] is the first added layer.


# In[5]:


net[0].weight, net[0].bias # use properties weight and bias to access


# In[6]:


net[0].weight.data(), net[0].bias.data() # use function data to get initialization ndarray


# In[7]:


net[0].weight.grad()
# using function 'grad' to get the gradient ndarray of parameters.
# Here without backward pass calculation, all elements in gradient ndarray are still zero. 


# In[8]:


net[1].bias.data() # the second added layer


# In[9]:


net.collect_params()
# Using function 'collect_params', we can get all parameters in this multilayer percetorn.


# In[10]:


net.collect_params('.*weight') # get all weight parameters
# It's easy to see that all weight parameters is end with 'weight' generally,
# such as dense0_weight and dense1_weight.
# Using regex could match them.


# # Share Model Parameters

# In[14]:


net = nn.Sequential()
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
       shared,
       nn.Dense(8, activation='relu', params=shared.params),
       nn.Dense(10))

net.initialize()

X = nd.random.uniform(shape=(2, 20))
Y = net(X)

net[1].weight.data() == net[2].weight.data()
# The second layer and the third layer share same weight parameters.
# And there's no doubt that the bias is same as well.


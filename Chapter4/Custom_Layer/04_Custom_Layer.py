#!/usr/bin/env python
# coding: utf-8

# # Custom Layer

# ## Custom Layer Without Model Parameters

# In[1]:


from mxnet import nd, gluon
from mxnet.gluon import nn

class CenteredLayer(nn.Block): # must inherited by class Block
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
        
    def forward(self, x):
        return x - x.mean()


# In[2]:


layer = CenteredLayer()
layer(nd.array([1, 2, 3, 4, 5]))


# In[3]:


net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())


# In[4]:


net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
y.mean().asscalar()


# ## Custom Layer With Model Parameters

# In[8]:


params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params


# In[9]:


class MyDense(nn.Block):
    # uints is the number of output, and in_uints is the number of input
    def __init__(self, uints, in_uints, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_uints, uints))
        self.bias = self.params.get('bias', shape=(uints,))
        
    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data() # linear regression
        return nd.relu(linear) # activation function 'relu'


# In[26]:


dense = MyDense(uints=3, in_uints=5)
dense.params


# In[28]:


dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))


# In[34]:


net = nn.Sequential()
net.add(nn.Dense(8, in_units=64),
       MyDense(1, in_uints=8)) # custom data, which is like `nn.Dense(1, in_uints=8, activation='relu')`
net.initialize()
net(nd.random.uniform(shape=(2, 64)))


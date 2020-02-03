#!/usr/bin/env python
# coding: utf-8

# # Deferred Initialization of Model Parameters
# When using gluon and call function `initialize`, it seems that we have initialized all paramters. But we don't specify the dimension of input data, so the weight array is unsure. Therefore, **it is impossible to have initialized at that point**.

# Here implementing class `MyInit` to show the deferred initialzation of model parameters.

# In[2]:


from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
       nn.Dense(10))
net.initialize(init=MyInit())


# In[3]:


X = nd.random.uniform(shape=(2, 40))
Y = net(X) # model initialize at this point


# In[5]:


Y = net(X) # not initialize


# From above output, we can get that only when gluon konw enough model information can it initialize the model parameters.
# Each coin has two side. It means that we cannot use `data` or `set_data` function to change model parameters before the first forward pass calculation.

# ## Avoid Deferred Initialization

# ### First, reinit for initialized model.

# In[ ]:


net.initialize(init=MyInit(), force_reinit=True)


# ### Second, Specify Input Unit When Creating Model

# In[6]:


net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'),
       nn.Dense(10, in_units=256))

net.initialize(init=MyInit())


#!/usr/bin/env python
# coding: utf-8

# ## Multilayer Perceptron From Scratch

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet.gluon import loss as gloss
from mxnet import nd
import utils


# ### Read Datasets

# In[3]:


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)


# ### Define Model parameters

# In[4]:


num_input, num_hidden, num_output = 784, 256, 10 # the unit number of input, hidden and output layer
W1 = nd.random.normal(scale=0.01, shape=(num_input, num_hidden)) # W1 is in R784*256
b1 = nd.zeros(num_hidden)
W2 = nd.random.normal(scale=0.01, shape=(num_hidden, num_output)) # W2 is in R256*10
b2 = nd.zeros(num_output)

params = [W1, b1, W2, b2]

# attach gradient
for param in params:
    param.attach_grad()


# ### Define Activation Function
# $$ReLU(x) = \max(x, 0)$$

# In[5]:


def relu(x):
    return nd.maximum(x, 0) # x.relu()


# ### Define Model
# $$O = HW_2 + b_2 = \phi(XW_1 + b_1)W_2 + b_1$$
# Here activation function $\phi$ is **ReLU**. 

# In[6]:


def net(X):
    X = X.reshape((-1, num_input))
    H = relu(nd.dot(X, W1) + b1) # \phi
    return nd.dot(H, W2) + b2 # HW_2 + b_2


# ### Define Loss Function

# In[7]:


loss = gloss.SoftmaxCrossEntropyLoss()


# ### Train Model

# In[9]:


num_epoch, lr = 5, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, params, lr)
# epoch, loss 0.3799, train acc 0.859, test acc 0.872 
# It seems that multilayer perceptron is better, because we add hidden layer and activation function
# Before the test accuracy is 0.84 or 0.85 at most!


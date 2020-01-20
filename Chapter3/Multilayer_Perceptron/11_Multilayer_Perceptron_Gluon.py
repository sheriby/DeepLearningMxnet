#!/usr/bin/env python
# coding: utf-8

# ## Multilayer Perceptron Implemented By Gluon

# In[3]:


from mxnet.gluon import loss as gloss, nn # nn is neural network
from mxnet import gluon, init # init is to initialize the weight and bias
import d2lzh as d2l
import utils


# ### Define Model

# In[2]:


net = nn.Sequential() # it is a container
net.add(nn.Dense(256, activation='relu'), nn.Dense(10)) # use relu as activation function
net.initialize(init.Normal(sigma=0.01)) # initialize the weight parameters


# ### Train Model

# In[6]:


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size) # get data iter

loss = gloss.SoftmaxCrossEntropyLoss() # git softmax cross entropy loss function
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5}) # get stochastic gradient descent

num_epochs = 5 # number of iterations
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, trainer=trainer) # start training


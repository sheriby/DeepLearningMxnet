#!/usr/bin/env python
# coding: utf-8

# ### Concise Softmax Regression
# Use gluon to implement softmax regression is also concise

# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
import utils
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn


# #### Read Datasets

# In[30]:


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)


# #### Define And Initialize Model

# In[31]:


net = nn.Sequential()
net.add(nn.Dense(10)) # 10 output(catagroies) in output layer
# Use normal distribution with mean difference of 0 and standard deviation of 0.01 to initialize weight
net.initialize(init.Normal(sigma=0.01)) 


# #### Softmax And Corss Entropy Loss Function

# In[32]:


loss = gloss.SoftmaxCrossEntropyLoss()


# #### Define Optimization Algorithm

# In[33]:


trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.1})


# #### Training Model

# In[34]:


num_epoch = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch,
              batch_size, None, None, trainer)


# ### Exercise
# Try adjusting hyperparameters, such as batch

# > Answer: I try 10, 100, 256, 500, it seems that the distinction is slight.

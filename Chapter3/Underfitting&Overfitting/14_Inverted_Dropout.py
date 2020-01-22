#!/usr/bin/env python
# coding: utf-8

# # Inverted Dropout
# Like `Weight Delay`, `Inverted Dropout` is a efficient method to slove the problem of `Underfitting and Overfitting` as well.

# ## Method
# Given a multilayer perceptron with one hidden layer, the number of input unit is $5$ and the number of hidden unit. Among them, $h_i$ is i-th hidden unit.
# $$h_i = \phi(x_1w_{1i} + x_2w_{2i} + x_3w_{3i} + x_4w_{4i} + b_i)$$
# When using dropout in this hidden layer, the hidden unit will be **dropped out with a certain probability $p$**. If not dropped out, it will be **stretched divided by $1- p$**.  
# Assume $\xi_i$ is $0$ or $1$ with the probability $p$ and $1-p$. Using dropout method, we can get that  
# $$h_i' = \frac{\xi_i}{1-p}h_i$$
# If $\xi$ is $0$, then $h_i' = 0$.  
# If $\xi$ is $1$, then $h_i' = \frac{h_i}{1-p}$.
# Due to $E(\xi_i) = 1 - p$, we can get  
# $$E(h_i') = \frac{E(\xi_i)}{1-p}h_i = h_i$$
# We will not change the maths expectation using the dropout method. The dropout method functions as **Regularization**. It is a efficient method to **handle the overfitting** when training datasets, but usually we don't use dropout when testing datasets to get more accurate result.

# ## Implement from Scratch

# In[10]:


import d2lzh as d2l
import utils
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn


# In[6]:


def dropout(X, drop_prob): # 'prob' is probability
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # Drop all elements in this condition
    if keep_prob == 0:
        return X.zeros_like()
    mask = nd.random.uniform(0, 1, X.shape) < keep_prob # nice code here. 
    # some element will be zero here, and others will be one, which is determined by keep_prob
    return mask * X / keep_prob


# In[3]:


X = nd.arange(16).reshape((2, -1))
dropout(X, 0) # None elements will be dropped out.


# In[4]:


dropout(X, 0.5) # About 8 elements will be dropped out.


# In[7]:


dropout(X, 1) # All elements will be dropped out


# ## Define Model Parameters
# Take Fashion-MNIST as example. Define a multilayer perceptron with two hidden layers, and there are 256 unit in each hidden layer.

# In[9]:


num_inputs, num_hidden1, num_hidden2, num_outputs = 784, 256, 256, 10

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hidden1))
W2 = nd.random.normal(scale=0.01, shape=(num_hidden1, num_hidden2))
W3 = nd.random.normal(scale=0.01, shape=(num_hidden2, num_outputs))

b1 = nd.zeros(num_hidden1)
b2 = nd.zeros(num_hidden2)
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()


# ## Define Model
# Use dropout method only when we are in training mode.

# In[16]:


drop_prob1, drop_prob2 = 0.2, 0.5

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training(): # if in training mode
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3


# ## Training and Testing Model

# In[17]:


num_epoch, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, params, lr)

# test accuracy is higher than train accuracy, great!


# ## Implement Consicely by Gluon

# In[19]:


net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob1), # add dropout layer after the first fully connection layer 
        nn.Dense(256, activation='relu'),
        nn.Dropout(drop_prob2), # add dropout layer after the second fully connection layer
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoch, batch_size, None, None, trainer)


# ## Exercise
# 1. What if we reverse the position of probability hyperparameters?

# > Answer: 

# 2. Add number of epoch, observe the distinction of using dropout or not.

# > Answer:

# 3. Complex the model or add more hidden layer unit, will the effect of dropout more obvious?

# > Answer:

# 4. Compare dropout to weight delay. If simultaneously using dropout and weight delay, what will happen?

# > Answer:

#!/usr/bin/env python
# coding: utf-8

# # Weight Delay
# Although increse the size of training dataset is a good way to solve underfitting and overfitting question, but usually it is difficult to get more data.  
# `Weight Delay` is efficient way to solve underfitting and overfitting question.             

# ## Method
# `Weight Delay` is equivalent to `L2 Norm Regularization`.  
# L2 norm regularization **add a penalty square item** in loss function to get minimum function.
# For example, linear regression loss function is as follow.
# $${\scr l}(w_1, w_2, b) = \frac1n\sum_{i=1}^n\frac12(x_1^{(i)}w_1 + x_2^{(i)}w_2 + b - y^{(i)})^2$$
# New loss function with L2 norm penalty item is
# $${\scr l}(w_1, w_2, b) + \frac\lambda{2n} \parallel\!w\parallel ^2$$
# $\parallel w\parallel$ is L2 norm(square root of square sum) of $w$. ($\parallel w \parallel = \sqrt{w_1^2 + w_2^2}$)  
# Among them, the hyperparamter $\lambda \gt 0$. When all weight parameter is zero, the penalty is smallest. **The weight parameter is bigger, the penalty is bigger.**  
# 
# $$w_1 \leftarrow \left(1 - \frac{\eta\lambda}{{\frak |B|}}\right)w_1 - \frac{\eta\lambda}{{\frak |B|}}\sum_{i \in {\frak B}}x_1^{(i)}(x_1^{(i)}w_1 + x_2^{(i)}w_2 - b - y^{(i)})$$
# $$w_2 \leftarrow \left(1 - \frac{\eta\lambda}{{\frak |B|}}\right)w_2 - \frac{\eta\lambda}{{\frak |B|}}\sum_{i \in {\frak B}}x_2^{(i)}(x_1^{(i)}w_1 + x_2^{(i)}w_2 - b - y^{(i)})$$

# ## High-dimensional Linear Regression Experiment
# $$ y = 0.05 + \sum_{i=1}^p0.01x_i + \epsilon$$

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

featrues = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(featrues, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_featrues = featrues[:n_train, :], featrues[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


# In[4]:


# test_featrues[0], test_featrues[0], train_labels[0], test_labels[0]


# ## Implement From Scratch

# ### Initialize Model Parameters

# In[29]:


def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


# ### Define L2 Norm Penalty Item

# In[13]:


def l2_penalty(w):
    return (w**2).sum() / 2


# ### Define Training and Testing

# In[30]:


batch_size, num_epoch, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features,
                        train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epoch):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_featrues, w, b),
                            test_labels).mean().asscalar())
    
    d2l.semilogy(range(1, num_epoch + 1), train_ls, 'epochs', 'loss',
                range(1, num_epoch + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())


# ### Observe Overfitting

# In[31]:


fit_and_plot(lambd=0)
# train error is small but test erros is very big.


# ### Use Weight Delay

# In[32]:


fit_and_plot(lambd=3)
# It is easy to see that though train error become bigger, train error is closer to zero.


# ## Implement Consicely

# In[40]:


def fit_and_plot_gluon(wd): # wb is weight delay
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    
    # weight delay for weight parameters which is often end with 'weight'
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate' : 0.003, 'wd' : wd})
    # not weight dealy for bias parameters which is often end wight '.bias'
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate' : 0.003})
    
    train_ls, test_ls = [], []
    for _ in range(num_epoch):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # call step two trainer for weight and bias
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
            
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_featrues), test_labels).mean().asscalar())
    
    d2l.semilogy(range(1, num_epoch + 1), train_ls, 'apoch', 'loss',
                range(1, num_epoch + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())


# In[41]:


fit_and_plot_gluon(0)


# In[42]:


fit_and_plot_gluon(3)


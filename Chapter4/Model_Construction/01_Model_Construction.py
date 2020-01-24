#!/usr/bin/env python
# coding: utf-8

# # Model Construction
# It Makes it flexible to construct model that using model construction method based by `Class Block`.

# ## Construct Model by Inheriting Class Block

# In[1]:


from mxnet import nd
from mxnet.gluon import nn

# construct a mutlilayer perceptron
class MLP(nn.Block): # inherit Block
    # declare layer with model parameters.
    # Here declare two fully connection layer
    def __init__(self, **kwargs):
        # use the contruction function of super class Block to do some
        # necessary initialization. 
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu') # hidden layer
        self.output = nn.Dense(10) # output layer
    # define the forward pass calculation of model.
    # That means how to calculate output result using the input x
    def forward(self, x): # this function shall be named as 'forward'
        return self.output(self.hidden(x))


# We needn't write 'backward' function in above MLP because the model will automaticly generate `backward` function by using `autograd`.  
# We can get model varialbe `net` by instantiate class `MLP`.Among them, `net(X)` will call function `__call__` inherited by class `Block`, which will call function `forward` in class MLP to do froward pass calculation.

# In[2]:


X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
net(X)


# class `Block` is a freely createable components. It's subclass can be a layer such as `Dense`, a model such as above `MLP`, or something like a part of model else.

# ## Class Sequential is Inherited by Class Block
# Here we will implement `MySequential` to simply show how class `Sequential` work.

# In[12]:


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)
    
    def add(self, block):
        # store block variable in _children which is a `OrderedDict` and will be call when we
        # instantiate MySequential 
        # property _children and name is inherited from class Block
        self._children[block.name] = block
    
    def forward(self, X): # forward pass calculate
        # OrderedDick will make sure that we will traverse it 
        # in the order of how they are added.
        for block in self._children.values():
            X = block(X)
        return X


# In[13]:


net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)


# ## Constructe Complex Model
# Here we will implement a complex model `FancyMLP` inherited by class `Block`.

# In[14]:


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # rand_weight parameter in get_constant function will not be 
        # changed. It is constant.
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')
        
    def forward(self, X):
        X = self.dense(X)
        # use constant rand_weight parameter
        X = nd.relu(nd.dot(X, self.rand_weight.data()) + 1)
        # reuse fully connectional layer
        X = self.dense(X)
        # control flow
        while X.norm().asscalar() > 1:
            X /= 2
        if X.norm().asscalar() < 0.8:
            X *= 10
        return X.sum()


# In[17]:


net = FancyMLP()
net.initialize()
net(X)


# In[18]:


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                    nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')
        
    def forward(self, x):
        return self.dense(self.net(x))


# In[19]:


net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())

net.initialize()
net(X)


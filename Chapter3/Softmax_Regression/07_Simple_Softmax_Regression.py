#!/usr/bin/env python
# coding: utf-8

# ### Implement Softmax Regression From Scratch

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
import utils
from mxnet import autograd, nd


# #### Read Datasets

# In[2]:


batch_size = 256
train_iter, test_iter = utils.load_data_fashion_mnist(batch_size) # see details in utils.py and 06


# In[3]:


num_inputs = 784 # 28 * 28. each pixel is a feature.
# transform a 28 * 28 matrix to a 1 * 784 vector, which may lead to some question because we lose the
# vertical information.
num_outputs = 10 # there are 10 catagory

# initialize weight and bias
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs)) # W is in R784*10
b = nd.zeros(num_outputs) # initialize the bias as zero. b is in R10

# attach gradient
W.attach_grad()
b.attach_grad()


# In[8]:


# for X, y in train_iter:
#     utils.show_fashion_mnist(X, utils.get_fashion_mnist_labels(y.asnumpy())) # numpy is better maybe,
#     break


# #### Implement Softmax Operation

# In[4]:


X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True) # sum in column and sum in row


# In[5]:


def softmax(X): # X is R784(features)*256(batch size)
    X_exp = X.exp() # each element x in matrix X will be converted to exp(x)
    partition = X_exp.sum(axis=1, keepdims=True) # sum in row
    return X_exp / partition # boradcast => each element divided by the sum of its row!


# In[6]:


X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1) # make every number positive between 0~1 and the sum of each row is 1


# #### Define Model
# $$O = XW + b$$

# In[7]:


def net(X): # X in R256*1*28*28, W in R784*10, b in R10, X -> reshape 256*784, XW ->256*10
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# #### Define Cross Entropy Loss Function
# $${\scr l}(\Theta) = \frac1n\sum_{i=1}^nH(y^{(i)}, \hat y^{(i)}) = -\frac1n\log \hat y_k^{(i)}\left(Among\ them\ y_k^{(i)} = 1\right)$$

# In[8]:


y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2], dtype='int32')
nd.pick(y_hat, y) # it is same as `y_hat.pick(y)` 
# Functin `pick` => pick element from y_hat by using y as index array


# In[9]:


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log() # it is positive because logX is negative( 0 < x < 1)


# #### Calculate Classification Accuracy

# In[10]:


def accuracy(y_hat, y):
    # Function argmax return the max index array in specified axis
    # axis=1 is max index in a row
    # Function mean return average value of ndarray
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar() # return right quantities


# In[22]:


test = nd.random.uniform(1, 100, shape=(4, 4))
test


# In[23]:


test.argmax(axis=0), test.argmax(axis = 1)
# argmax function seems that only support by 'float32'
# axis=0 => in column
# axis=1 => in row


# In[11]:


accuracy(y_hat, y) # the first prediction is wrong and the second is right so the accuracy is 0.5(50%)


# In[12]:


def evaluate_accuracy(data_iter, net):
    acc_num, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        acc_num += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size # y.size is batch size (256)
    return acc_num / n # test model accuracy


# In[13]:


evaluate_accuracy(test_iter, net)


# #### Training Model

# In[28]:


num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params = None, lr = None, trainer = None):
    for epoch in range(num_epochs):
        train_l_num, train_acc_num, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None: # if there is no trainer, use sgd trainer model
                d2l.sgd(params, lr, batch_size)
            else: # use gluon.Trainer()
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_num += l.asscalar()
            train_acc_num += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net) # use testing set to evaluate this model
        print('epoch %d, loss: %.4f, train accuracy: %.3f, test accuracy: %.3f'
             %(epoch + 1, train_l_num / n, train_acc_num / n, test_acc))
        
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


# In[31]:


for X, y in test_iter:
    break

true_labels = utils.get_fashion_mnist_labels(y.asnumpy())
pred_labels = utils.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

utils.show_fashion_mnist(X[0:9], titles[0:9])


# ### Exercise
# 1. If we implement softmax according to its maths definition, what will happen? (Tip: try to calculate exp(50)

# > Answer: exp(50) is very big, if we implement softmax based on its maths defintion, our pragram will throw error or get unstable calculation results.

# 2. Function `cross_entropy` implements according to its maths definition. Is there anything with this implement?(Tip: thinking about the domain of a logarithmic function)

# > Answer: The domain of logarithmic function is from 0 to infinity. So we must keep all feature positive and keeping them between 0-1 is the best choice. 

# 3. What methods can you think of to solve the above problems?

# > Answer: None, How stupid I am!

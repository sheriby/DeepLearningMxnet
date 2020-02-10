#!/usr/bin/env python
# coding: utf-8

# # Deep Convolutional Neural Netword AlexNet

# ## AlexNet Model

# In[54]:


import d2lzh as d2l
from mxnet import nd, init, gluon
from mxnet.gluon import nn, data as gdata
import os
import sys
import mxnet as mx


# In[94]:


net = nn.Sequential()

net.add(nn.Conv2D(24, kernel_size=5, strides=2, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2),
       nn.Conv2D(64, kernel_size=5, padding=2, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2),
       nn.Conv2D(96, kernel_size=2, padding=1, activation='relu'),
       nn.Conv2D(96, kernel_size=3, padding=1, activation='relu'),
       nn.Conv2D(96, kernel_size=3, padding=1, activation='relu'),
       nn.MaxPool2D(pool_size=3, strides=2),
       nn.Dense(192, activation='relu'), nn.Dropout(0.5),
       nn.Dense(192, activation='relu'), nn.Dropout(0.5),
       nn.Dense(10))


# In[95]:


X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# ## Read Dataset

# In[13]:


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join(
        '~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnisi_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnisi_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_worker = 0 if sys.platform.startswith('win') else 4
    train_iter = gdata.DataLoader(
        mnisi_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_worker
    )
    test_iter = gdata.DataLoader(
        mnisi_test.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_worker
    )
    return train_iter, test_iter


# In[96]:


batch_size = 1
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=None)


# ## Train Model

# In[97]:


lr, num_epochs, ctx = 0.001, 5, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=mx.gpu(), init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, mx.gpu(), num_epochs)


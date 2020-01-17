#!/usr/bin/env python
# coding: utf-8

# ### 线性回归的简洁实现
# 我们可以使用`MXNet`提供的`gluon`接口简洁的实现线性回归。

# #### 生成数据集

# In[3]:


from mxnet import autograd, nd

num_input = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2

features = nd.random.normal(scale=1, shape=(num_example, num_input))
lables = features[:, 0] * true_w[0] + features[:, 1] * true_w[1] + true_b
lables += nd.random.normal(scale=0.01, shape=lables.shape)
# 以上步骤和之前的是相同的


# #### 读取数据集

# In[4]:


from mxnet.gluon import data as gdata

batch_size = 10
# 将训练数据的特征和标签结合
dataset = gdata.ArrayDataset(features, lables)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True) # 第三个参数为True表示随机读取


# In[5]:


for X, y in data_iter:
    print(X, y)
    break


# #### 定义模型

# In[6]:


from mxnet.gluon import nn # nn 表示 neural networks(神经网络)

net = nn.Sequential() # 一个串联各层的容器
net.add(nn.Dense(1)) # 线性回归中的全连接层Dense， 输出的个数为1
# 我们无需指定每一层的输入的形状，后面执行的时候会自动推断。


# #### 初始化模型参数

# In[7]:


from mxnet import init

net.initialize(init.Normal(sigma=0.01)) # 初始化方法，将权重设置为均值为1，标准差为0.01的正态分布，偏差设置为0。


# #### 定义损失函数

# In[8]:


from mxnet.gluon import loss as gloss

loss = gloss.L2Loss() # 平方损失，又被称为L2范数损失


# #### 定义优化算法

# In[9]:


from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate' : 0.03}) # 定义优化算法和学习率
# sgd => stochastic gradient descent 随机梯度下降法


# #### 训练模型

# In[10]:


num_epochs = 5 # 迭代的次数。
for epoch in range(num_epochs):
    for X, y in data_iter: # 获取一批数据
        with autograd.record():
            l = loss(net(X), y) # 计算损失函数
        l.backward() # 求导
        trainer.step(batch_size) # 梯度下降
    l = loss(net(features), lables) # 计算模型的loss
    print('epoch %d, loss: %f'%(batch_size + 1, l.mean().asscalar()))


# In[11]:


dense = net[0] # 获取训练得到的模型参数，因为输出层只有一个节点，所以就是net[0]
true_w, dense.weight.data() # 获取训练得到的权重


# In[12]:


true_b, dense.bias.data() # 获取训练得到的偏差


# ### 练习
# 1. 如果将`l = loss(net(X), y)`替换成`l = loss(net(X), y).mean()`，我们需要将`trainer.step(batch_size)`相应地改成`trainer.step(1)`。这是为什么呢？

# > 答： 那就要先观察一下官方文档中这个函数的定义了。  
# > `step（batch_size，ignore_stale_grad = False)`  
# > `Makes one step of parameter update. Should be called after autograd.backward() and outside of record() scope.`  
# > `batch_size (int) – Batch size of data processed. Gradient will be normalized by 1/batch_size. Set this to 1 if you normalized loss manually with loss = mean(loss).`  
# 看来官方文档中已经说的足够清楚了。我们`l.backward()`相当于`l.sum().backward()`,指的是一批样本中的所有的数据的损失的梯度，我们需要传递`batch_size`，函数的内部将`/batch_size`求均值以弱化更新参数带来的影响。
# 如果我们先对其进行了`mean()`操作，操作完了之后变成了$\mathbb{R}$的一个标量，此时就不必除以`batch_size`了，将这个值设置为`1`就行了。

# 2. 查阅`MXNet`的文档，看看`gluon.loss`和`init`模块中提供了哪些损失函数和初始化方法。

# > 答:  
# > - `gluon.loss`中有很多的损失函数。  
# >   - `L1Loss`
# $$L = \sum_i|label_i - pred_i|$$
# >   - `L2Loss`
# $$L = \frac12\sum_i|label_i - pred_i|^2$$
# >   - `SigmoidBinaryCrossEntropyLoss`
# $$prob = \frac1{1 + e^{-pred}}$$
# $$L = - \sum_ilable_i*log(prob_i) + pos\_weight + (1 - lable_i)*log(1 - prob_i)$$
# >   - `SoftmaxCrossEntropyLoss`
# $$p = softmax(pred)$$
# $$L = -\sum_i\log p_{i,\ lable_i}$$
# >   - `KLDivLoss`
# $$L = \sum_ilable_i*[\log(lable_i) - \log(pred_i)]$$
# >   - `LogisticLoss`
# $$L = \sum_i\log\left(1 + e^{-pred_i * lable_i}\right)$$
# > 还有非常多的损失函数，但是我基本上都看不懂，在这儿抄一遍也没啥意思，打公式还是蛮费劲的。  
# >  
# > - `init`中有很多初始化的方法。
# >    - `Normal`  *Initializes weights with random values sampled from a normal distribution with a mean of zero and standard deviation of sigma.*
# >    - `Load`    *Initializes variables by loading data from file or dict.*
# >    - `LSTMBias`    *Initialize all biases of an LSTMCell to 0.0 except for the forget gate whose bias is set to custom value.*
# >    - `MSRAPrelu`   *Initialize the weight according to a MSRA paper.*
# >    - `Mixed`    *Initialize parameters using multiple initializers.*
# >    - `One`    *Initializes weights to one.*
# >    - `Orthogonal`  *Initialize weight as orthogonal matrix.*
# >    - `Uniform`  *Initializes weights with random values uniformly sampled from a given range.*
# >    - `Xavier`   *Returns an initializer performing “Xavier” initialization for weights.*
# >    - `Zero`   *Initializes weights to zero.*
# >    - `register`    *Registers a custom initializer.*  
# >上面就是官方文档中的所有的初始化的方法。虽然大多数我都不知道是什么意思~~

# 3. 如何访问`dense.weight`的梯度？

# > 答: 通过`dense.weight.grad()`可以访问`dense.weight`的梯度。(瞎试出来的，我先看的`dense.weight.grad`，输出告诉我这是一个函数，那么就调用这个函数就完事了)

# In[17]:


dense.weight.grad()


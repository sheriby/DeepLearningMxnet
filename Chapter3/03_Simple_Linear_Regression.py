#!/usr/bin/env python
# coding: utf-8

# ### 线性回归的简答实现
# 如何只利用`NDArray`和`autograd`来实现一个线性回归的训练。

# In[15]:


# 作图设置为嵌入显示
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython import display
from matplotlib import pyplot as plt
from mxnet import nd, autograd
import random


# #### 生成数据集
# 根据一个给定的真实目标生成一个随机的数据集。

# In[16]:


num_input = 2 # 样本的特征数量
num_example = 1000 # 样本的数量
true_w = [2, -3.4] # 真实的权重
true_b = 4.2 # 真实的偏移量

features = nd.random.normal(scale=1, shape=(num_example, num_input)) # 均值为0， 标准差为1的随机特征 R1000*2
lables = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b # 生成一千个标签。 R1000*1
lables += nd.random.normal(scale=0.01, shape=lables.shape) # 添加噪声，使得数据并不是完全线性分布。


# In[3]:


features[0], lables[0]


# In[17]:


def use_svg_display():
    # 使用矢量图显示
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize
    
set_figsize()
plt.scatter(features[:, 1].asnumpy(), lables.asnumpy(), 1, c='pink');# 加分号只显示图
plt.scatter(features[:, 0].asnumpy(), lables.asnumpy(), 1, c='skyblue'); # 第三个参数用来控制散点的大小。


# #### 读取数据集
# 在训练数据的时候，我们需要遍历数据集并不断读取小批量数据样本。

# In[18]:


def data_iter(bath_size, features, lables):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices) # 使得样本的读取是随机的
    for i in range(0, num_example, bath_size):  # 读取一个批量的数据
        j = nd.array(indices[i: min(i + bath_size, num_example)])
        yield features.take(j), lables.take(j)


# In[19]:


batch_size = 10

for x, y in data_iter(batch_size, features, lables):
    print(x, y)
    break


# #### 初始化模型参数

# In[20]:


w = nd.random.normal(scale=0.01, shape=(num_input, 1))
b = nd.zeros(shape=(1,))


# 后面需要使用自动求梯度来迭代参数的值，所以我们要先来创建他们的梯度

# In[21]:


w.attach_grad()
b.attach_grad()


# #### 定义模型
# 线性回归的模型为$y = Xw + b$

# In[22]:


def linreg(X, w, b):
    return nd.dot(X, w) + b


# #### 定义损失函数
# 损失函数为${\scr l}^{(i)}(w_1,\ w_2,\ b) = \frac12(\hat y^{(i)} - y^{(i)})^2$

# In[23]:


def squared_lost(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # 为了确保二者的形状是一致的


# #### 定义优化算法
# 不断的迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。
# $$\theta \leftarrow \theta - \frac\eta{\frak |B|}\sum_{i \in {\frak B}}\nabla_\theta{\scr l}^{(i)}(\theta)$$

# In[24]:


def sgd(params, lr, batch_size):
    # params = [w, b]
    for param in params:
        param[:] = param - lr / batch_size * param.grad


# #### 训练模型

# In[39]:


lr = 0.03 # learning rate
num_epochs = 5
net = linreg
loss = squared_lost

for epoch in range(num_epochs):
    # 在每个迭代周期中，会使用训练数据集中的样本一次。
    # X和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, lables):
        with autograd.record():
            l = loss(net(X, w, b), y) # l是有关小批量X和y的损失 R10*1
        l.backward() # 对损失函数进行求导 这里的l不是变量。 l.backward() => l.sum().backward()
        sgd([w, b], lr, batch_size) # 进行梯度下降
    train_l = loss(net(features, w, b), lables) # 这里的w，b是训练之后的w和b, train_l是对一千个样本数据来说的损失
    # train_l R1000*1
    print('epoch %d, loss %f'%(epoch + 1, train_l.mean().asscalar())) # mean是计算平均值


# In[28]:


w, true_w


# In[30]:


b, true_b


# #### 练习
# 1. 为什么`squared_loss`函数中需要使用`reshape`?

# > 答: `squared_loss(y_hat, y)`。其中`y_hat`是通过模型预测的理论值，而`y`是真实值。上面共有两处调用了`loss`函数，第一个参数都是`net`函数的返回值。  
# > 先来看第一块——`loss(net(X, w, b), y)` => 这里的`net(X, w, b)`就是$Xw + b \in \mathbb{R}^{10 \times 1}$，$y$则是由一个生成器产生，$y \in \mathbb{R}^{1 \times 10}$。二者的形状并不一样，所以不可以进行相加的操作。如果想要进行相加的操作，需要将$y$进行`reshape`。  
# > 第二次使用——`loss(net(features, w, b), lables)` => 这里的两个参数都是$\mathbb{R}^{1000 \times 1}$，是无所谓是否`reshape`的。

# 2. 尝试不同的学习率，观察损失函数的下降快慢。

# > 答: 当学习率很低的时候，我们需要很长的时间才能够使得损失函数收敛。但是当学习比较大的时候，损失函数将不再收敛，而是扩散，主要是因为我们在梯度下降的时候下降的太过剧烈跨过了极值点。我们要选取的是一个适当的学习率，既不能太大也不能太小。

# 3. 如果样本个数不能被批量大小整除， `data_iter`函数的行为会有什么变化？

# > 答: 上面我们只调用了五次`data_iter`函数，也就是只使用了五十个数据就预测出了很好的模型。（其实这个例子中两三次基本上就收敛了）一般情况下，我们不会将样本中的所有的数据都使用玩的，所以不必担心是否整除的问题。如果我们发现不能整除了，那唯一的变化就是最后一次不会获取到`batch_size`这么多的数据了，而是`num_example % batch_size`这么多的数据，因为`data_iter`中有如下的代码段，`min(i + batch_size, num_example)`。

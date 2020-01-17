#!/usr/bin/env python
# coding: utf-8

# ### 创建NDArray

# 首先从`MXNet`中导入模块`nd`，其中`nd`是`ndarray`的简称。使用的方法和`numpy`中的`ndarray`是相似的，但是鉴于`MXNet`中很多的特性，`MXNet`中的`nd`更适合做深度学习。

# In[1]:


from mxnet import nd


# 使用`arange`函数可以创建一个行向量。  
# 我们在for循环中使用的range时类型列表中的一个结构，nd也可以通过列表创建行向量。  
# 这里二者是相似的。

# In[2]:


x = nd.arange(12)
x


# In[3]:


y1 = nd.array(range(12))
print(y1)
# 通过列表创建ndarray
y2 = nd.array([1, 2, 3, 4])
print(y2)
# 通过多维数组创建矩阵
y3 = nd.array([[1, 2], [3, 4]])
print(y3)


# #### 查看NDArray的形状大小等信息  
# - 使用`shape`查看`ndarray`的维度。  
# - 使用`reshape`改变`ndarray`的维度。  
# - 使用`size`可以查看`ndarray`的大小。

# In[4]:


x.shape
# (12,)相当于(12, 1) => 表示这是一个行向量。


# In[5]:


x = x.reshape((1, 12)) # => 注意参数是一个元组
x = x.reshape(1, 12) # => 直接写也是可以的，那为什么要弄个元组呢？ 因为我们可以将其他ndarray的啥shape(元组)赋值给它。


# In[6]:


x.shape


# In[7]:


print(y2)
y2 = y2.reshape(y3.shape)
y2


# In[8]:


y2 = y2.reshape((-1, 4)); # 负数表示靠另一个维度进行推断。 原本是2*2, 有一个维度是4，另一个维度就可以推断出来是1了。
y2


# #### 一些特殊的矩阵
# 
# * 全0张量 
#     `nd.zeros((2, 3, 4))` => 创建一个维度为(2, 3, 4)的所有元素都是0的张量。 向量和矩阵都是特殊的张量。  
# * 全1张量
#     `nd.ones((1, 2))` => 和zeros相似
# * 随机张量
#     `nd.randonm.normal(0, 1, shape = (3, 4))`
#     创建一个维度为(3, 4)的均值为0， 标准差为1的张量

# In[9]:


nd.zeros((2, 3, 4))


# In[10]:


nd.ones((1, 2))


# In[11]:


nd.random.normal(0, 1, shape = (3, 4))


# #### NDArray的运算
# 
# - 按位加法 => `x + y`
# - 按位减法 => `x - y`
# - 按位乘法 => `x * y`
# - 按位除法 => `x / y`
# - 按位指数运算 => `x.exp()`
# - 矩阵的乘法 => `nd.dot(x, y)`
#   (矩阵的加法就是按位加法)

# In[12]:


x = nd.arange(16).reshape((4, 4))
x


# In[13]:


y = nd.ones((4, 4))
y


# In[14]:


x + y # 按位加法


# In[15]:


x * y # 按位加法，不是矩阵的乘法


# In[16]:


x / y # 按位除法


# In[17]:


y / x


# In[18]:


x.exp() # 指数运算， 每个元素x将变成 e^x


# In[19]:


nd.dot(x, y) # 矩阵的乘法，不再是按位乘法


# #### 连结多个`NDArray`  
# `nd.concat(x, y, dim=n)` 其中dim表示dimension的意思，表示在第几个维度上进行连结  
# 其中`dim = 0`表示在第一个维度上进行连结，也就是矩阵竖着摆放。 => 列数要相等  
# 同理`dim = 1`表示矩阵横着摆放。 => 行数要相等

# In[20]:


nd.concat(x, y, dim = 0), nd.concat(x, y, dim = 1)


# In[21]:


nd.concat(x, nd.ones((3, 8)).reshape(6, 4), dim=0)


# #### 条件判断
#  - 相等判断  
#      `x == y` 返回一个新的`ndarray`，相等的位置值为1，反之则为0。 
#  - 其他判断
#      同样的还有`>`, `<`, `>=`, `<=`这些判断。

# In[22]:


x == y, x >= y, x - 1 == y, x * 2 == y


# #### 求和操作和范数操作。
# 将张量中的所有的元素相加，返回一个(1, )的`ndarray`。  
# 如 `x.sum()`。
# 同样的有取范数的操作。  
# 所谓的L2就是将所有元素的平方和的平方根。  
# 如`x.norm()`。
# 返回的值并不是python中的一个数字，我们可以通过其他方式进行转换。
# #### 转为标量
# 使用`x.asscalar()`可以将(1, )的`ndarray`转为标量。
# 如`x.norm().asscalar()`

# In[23]:


x.sum(), x.norm()


# In[24]:


x.sum().asscalar(), x.norm().asscalar()


# ### 广播机制
# 当不同形状的`NDArray`做运算的时候，可能会触发广播机制。  
# 先复制元素使得两个`NDArray`形状相同的时候再做运算。  
# 
# 不是所有的情况都可以触发广播机制的，**必须要是行向量和列向量相加才行**  
# 如(3, 1) 和 (1, 4)相加之后的结果是(3, 4)。

# In[25]:


x = nd.arange(3).reshape(3, 1)
y = nd.arange(2).reshape(1, 2)
x, y


# In[26]:


x + y # 不同形状运算，触发了广播机制


# In[27]:


x = nd.arange(10).reshape(1, 2)
y = nd.arange(4).reshape(4, 1)
x , y


# In[28]:


x + y


# In[29]:


x * y


# ### 索引  
# 索引表示了元素的位置，python中的索引都是从0开始的。（和matlab和octave不一样，他们是从1开始的。  
# - x[1:3] 表示截取x的第二行和第三行。（索引的右面不包含）
# - x[:, 2:3] 表示截取x的第三行
# - x[1, 3] = 5 表示将第二行第四列的元素修改为5
# - x[1:2, :] = 12 表示将第二行的所有的元素都修改为12

# In[30]:


x = nd.arange(16).reshape(4, 4)
x


# In[31]:


x[1:3]


# In[32]:


x[:, 1:3]


# In[33]:


x[1, 3] = 5
x


# In[34]:


x[1:2, :] = 12
x


# In[35]:


x[:] = x - 1 # 在元素的本身进行修改，不创建额外的空间
x


# ### 运行的内存开销
# python中许多的操作都是新创建一块内存，即使是 `y = y + x`这种操作也是新创建一块内存z，然后将z赋值给y。  
# 非常的浪费内存。如果之后不必再使用到y，我们可以使用上面提到的这种方式。 `y[:] = y + x`或者 `y += x`

# In[36]:


x = nd.arange(12).reshape(3, 4)
y = nd.ones((3, 4))
x, y


# In[37]:


before = id(y)
y = y + x
id(y) == before # False


# In[38]:


z = nd.zeros_like(y) # => 相当于 nd.zeros(y.shape)
# z = nd.zeros(y.shape)
before = id(z)
z[:] = y + x
id(z) == before # True


# 但是实际上上面虽然是True，但是还是创建了额外的空间进行计算，然后覆盖了原来z的位置而已。如果想要真正的不创建任何的空间，
# 可以使用`nd.elemwise_add(x, y, out=z)`

# In[39]:


before = id(z)
nd.elemwise_add(x, y, out = z)
id(z) == before # True


# ### NDArray和Numpy的互相转换
# 通过`nd.array(p)`函数和`d.asnumpy()`函数可以将二者进行相互转换。其中p是`NumPy`实例，d是`NDArray`实例。

# In[40]:


import numpy as np


# In[41]:


p = np.ones((2, 3))
d = nd.array(p)
d


# In[42]:


d *= 2
p = d.asnumpy()
p


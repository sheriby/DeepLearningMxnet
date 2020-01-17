#!/usr/bin/env python
# coding: utf-8

# ## AutoGrad 自动求梯度

# #### 一个简单的例子

# In[1]:


from mxnet import autograd, nd


# In[2]:


x = nd.arange(4).reshape((4, 1))
x


# 为了求x的梯度，我们需要使用`x.attach_grad()`函数来申请存储梯度所需要的内存。

# In[3]:


x.attach_grad()


# 求梯度的操作比较消耗时间，为了减少计算和内存开销。我们需要在`autograd.record()`中进行梯度有关的计算。

# In[4]:


with autograd.record():
    y = 2 * nd.dot(x.T, x) # x.T表示x的转置矩阵


# 调用`y.backward()`函数进行自动求梯度。 需要注意的是，如果y不是一个标量，`MXNet`会将y中的元素求和得到新的变量，然后再对x求梯度。

# In[5]:


y.backward()


# 使用`x.grad`就可以得到相应的梯度。 如上面的例子中，如果x是一个数字，那么 `y = 2x²`，对x求梯度的结果就是`y = 4x`

# In[6]:


assert((x.grad - 4 * x).norm().asscalar() == 0) # assert断言操作，和c语言中使用方法相同
x.grad == 4*x


# #### 训练模式和预测模式
# 调用`autograd.record()`函数后，`MXNet`会记录并计算梯度。此外，`autograd`还会将运行模式从***预测模式***转为***训练模式***，可以通过`autograd.is_training()`进行查看。

# In[7]:


print(autograd.is_training()) # False
with autograd.record():
    print(autograd.is_training()) # True


# 在有些情况下，同一个模型在训练模式和预测模式下的表现并不相同，后面第三章会介绍二者的区别。

# #### 对python控制流进行求梯度
# 使用`MXNet`的一大好处是，即使在计算梯度的过程中包含了控制语句,（如if或者while）,也可以对变量进行自动求梯度。  
# 如以下的程序，循环的次数和c的值都是取决于我们输入的a是多少，此时我们依旧可以对a进行自动求梯度。

# In[8]:


def f(a):
    b = a * 2
    while b.norm().asscalar() < 1000:
        b = b * 2
    if b.sum().asscalar() > 0:
        c = b
    else:
        c = 100 + b
    return c


# In[9]:


a = nd.random.normal(shape=1) # 随机生成一个1*1的张量
print(a)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()


# 由于上面的函数的输出的结果肯定是`x*a`的形式或者`x*a + 100`, x取决于a的值。
# 当a为正值的时候， 那么对a求梯度的值就是x。而`x == c/a` 
# 当a为负值的时候， `x == (c-100)/a`，可以通过这个来验证自动求梯度是否正确。  

# In[10]:


assert((a.grad - c / a).norm().asscalar() == 0 or (a.grad - (c - 100) /a).norm().asscalar() == 0)
a.grad == c / a, a.grad, c / a, (c-100) / a


# ### 练习
# 1. 在上面的控制流求梯度的例子中，将a改成一个随机的向量或矩阵，此时计算结果c不再是标量，运行结果有何变化？试着分析运行的结果。

# In[11]:


a = nd.random.normal(shape=(3, 3))
print(a)
a.attach_grad()
with autograd.record():
    c = f(a)
c.backward()
print(a.grad, c/a)

# 相当于变成了对向量或矩阵的和进行求梯度。
b = a.sum()
print(b)
b.attach_grad()
with autograd.record():
    d = f(b)
d.backward()
b.grad, d/b, (d-100)/b


# 发现的结论，经过测试我发现一下的现象:
# - 变成向量或者矩阵之后梯度中所有位置的元素都相等，并且值不在等于`c/a`。
# - 如果我们将向量或者矩阵中所有的值相加得到标量，此时通过上面的方式再次求导，发现梯度就等于这时候求得的梯度。
# - 而且如果累加的值为负值的话，我们发现此时`d/b`的值还不等于求得的梯度了。
# 其中第三点，我发现是书上的错误，为负值的时候，梯度应该是`(d-100)/b`，上面已经做了说明。
# 第一点和第二点可以一起解释。  
# 之前我们说过，**如果y不是一个标量，MXNet会将y中的元素求和得到新的变量，然后再对x求梯度。**这里的y值得就是上面的c。此时c就不是标量。  
# 和标量相同，如果a中的元素之和大于0，(`a.sum().asscalar() > 0`)，得到的`c = xa`，此时这个x就是说要求的梯度。只不过这里的a和c变成了变量而已，别无两样。根据规则，此时求梯度要对c中进行求和然后再求梯度。求和变成了`x1*a1 + x2*a2 + x3*a3 ...`，其中`a1, a2, a3`是a中的元素。求导之后变成了`[x1, x2, x3...]`(格式应该和`a`相同)。`[x1, x2, x3...]`也就是`c/a`得到的`x`。
# 第二点所说的，将所有的元素相加得到`a1 + a2 + a3...`得到一个标量，此时和之前一样，c是一个标量，得到的梯度就是`c / (a1 + a2 + a3...)`，不过因为c就是一个标量，其值就是`x*a1 + x*a2 + x*a3 + ....`。  
# 上面虽然我写了`x1, x2, x3`，不过可以简单的看出他们是相等的，因此，两个方式得到的梯度也是相等的。

# 2. 重新测试一个控制流求梯度的列子， 运行并分析结果。

# In[12]:


def func(x):
    while x.norm().asscalar() < 100:
        x = x * 2
    if x.sum().asscalar() > 0:
        y = 2 * x * x
    else:
        y = x - 998
    return y


# In[13]:


x = nd.random.normal(shape=(3, 3))
print(x.sum())
x.attach_grad()
with autograd.record():
    y = func(x)
y.backward()
x.grad, 2 * y / x, (y + 998) / x


# 分析  
# 当x的求和大于0的话，此时`y = kx²`,梯度的值为`2kx`, 也就是`2 * y / x`。 
# 当x求和小于等于0的时候，此时`y = kx - 998`，梯度的值为`k`，也就是`(y + 998) / x`。

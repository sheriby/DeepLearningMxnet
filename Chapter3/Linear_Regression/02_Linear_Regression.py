#!/usr/bin/env python
# coding: utf-8

# ### 线性回归的表示方法
# ---
# #### 神经网络图
# 线性回归又被称为单层神经网络，只有输入层和输出层，没有隐藏层，而且输出层的神经元的个数只能有一个。这里的输出层直接和输入层相连，因此又被称为全连接层或者稠密层。
# 
# #### 矢量计算表达式
# 向量相加的两种方式

# In[1]:


from mxnet import nd
from time import time

a = nd.ones(shape = 1000)
b = nd.ones(shape = 1000)


# 1. 直接使用for循环

# In[2]:


start = time()
c = nd.zeros(shape = 1000)
for i in range(1000):
    c[i] = a[i] + b[i]
time() - start # 0.1661069393157959


# 2. 将两个向量直接做矢量加法

# In[3]:


start = time()
d = a + b
time() - start # 0.001001119613647461


# 由此可见，使用矢量加法的速度比for循环快了将近一百倍。我们应该尽量使用矢量运算，以提升计算的效率。  
# 回到之前的房屋预测的例子中来，如果我们对三个样本进行价格预测将得到。  
# 
# $$\hat y^{(1)} = x_1{(1)}w_1 + x_2{(1)}w_2 + b$$
# $$\hat y^{(2)} = x_1{(2)}w_1 + x_2{(2)}w_2 + b$$
# $$\hat y^{(3)} = x_1{(3)}w_1 + x_2{(3)}w_2 + b$$  
# 
# 现在我们要将三个等式转化成矢量运算。设:  
# 
# $$\hat y = \begin{bmatrix}y^{(1)} \\ y^{(2)} \\ y^{(3)} \end{bmatrix},\ 
# X = \begin{bmatrix}x_1^{(1)} & x_2^{(1)} \\ x_1^{(2)} & x_2^{(2)} \\ x_1^{(3)} & x_2^{(3)}\end{bmatrix},\ w = \begin{bmatrix}w_1 \\ w_2 \end{bmatrix} $$
# 
# 可以写成矢量运算，$\hat y = Xw + b$，其中加b使用广播机制。
# 
# 广义上来讲，当数据样本数为$n$， 特征数为$d$时，线性回归的矢量表达式为：
# $$\hat y = Xw + b$$
# 其中模型的输出$\hat y \in \mathbb{R}^{n \times 1}$，样本的特征$X \in \mathbb{R}^{n \times d}$，偏差$b \in \mathbb{R}$，相应的，批量数据的样本标签$y \in \mathbb{R}^{n \times 1}$。设模型参数$\theta = \begin{bmatrix}w1,& w2,& b\end{bmatrix}^\mathsf T$，我们可以重写损失函数为。  
# $${\scr l}(\theta) = \frac1{2n}(\hat y - y)^\mathsf T(\hat y - y)$$
# 小批量随机梯度下降的步骤可以相应的改写为
# $$\theta \leftarrow \theta - \frac\eta{\frak |B|}\sum_{i \in {\frak B}}\nabla_\theta{\scr l}^{(i)}(\theta)$$
# 其中
# $$\nabla_\theta{\scr l}^{(i)}(\theta) = \begin{bmatrix}\frac{\partial{\scr l}^{(i)}(w_1, w_2, b)}{\partial w_1} \\ \frac{\partial{\scr l}^{(i)}(w_1, w_2, b)}{\partial w_2} \\ \frac{\partial{\scr l}^{(i)}(w_1, w_2, b)}{\partial b} \end{bmatrix} = \begin{bmatrix}x_1^{(i)}(x_1^{(i)}w_1 + x_2^{(i)}w_2 + b - y^{(i)}) \\ x_2^{(i)}(x_1^{(i)}w_1 + x_2^{(i)}w_2 + b - y^{(i)}) \\ (x_1^{(i)}w_1 + x_2^{(i)}w_2 + b - y^{(i)})\end{bmatrix} = \begin{bmatrix}x_1^{(i)} \\ x_2^{(i)} \\ 1\end{bmatrix}(\hat y^{(i)} - y^{(i)})$$
# 上面使用的乘法的矩阵的按位乘法，因为两个矩阵都是$\mathbb{R}^{3 \times 1}$的矩阵。

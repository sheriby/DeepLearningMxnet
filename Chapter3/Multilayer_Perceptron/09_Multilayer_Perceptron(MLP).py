#!/usr/bin/env python
# coding: utf-8

# ## Multilayer Perceptron (MLP)

# ### Hidden Layer  
# ---
# ![](https://pic1.zhimg.com/80/v2-d87b0cd1f308baf1c554114ab5b56f24_hd.png)
# 
# `Input Layer` => `Hidden Layer` => `Hidden Layer` ... => `Output Layer`  
# There may be more than one hidden layer. And if there is none hidden layer, the output layer is also called as `Fully Connected Layer` or `Dense Layer`. what we learn before, such as linear regression or softmax regression, has none hidden layer.

# Specifically, given a mini-batch sample $X \in \mathbb R^{n \times d}$. The batch size is $n$, and the number of input is $d$.
# Suppose that there are just one hidden layer in multilayer perceptron, and the number of hidden unit is $h$. Record the output of hidden layer as $H$(It is also called as *hidden layer varible*). We can get $H \in \mathbb R^{1 \times h}$. Record the weight and bias of hidden layer as $W_o \in \mathbb R^{d \times h}$ and $b_k \in \mathbb R^{1 \times h}$. Similiarly, record the weight and bias of output layer as $W_o \in \mathbb R^{h \times q}$ and $b_o \in \mathbb R^{1 \times q}$.($q$ is the number of output)
# We can get following vector expression.
# $$H = XW_h + b_h$$
# $$O = HW_o + b_o$$
# Simultaneous above two formulas, we know that.
# $$O = (XW_h + b_h)W_o + b_o = XW_hW_o + b_hW_o + b_o$$
# Record $W_hW_o$ as a new weight matrix $W_x$, and record $b_hW_o + b_o$ as a new bias vector $b_x$. We can get.
# $$O = XW_x + b_x$$
# So this multilayer perceptron is equivalent to a sigle neural network without any hidden layer.It **make no sense**.

# ### Activation Function
# What leads to multilayer perceptron meaningless is that we just do `Affine Transformation` for data and superimposed affine transformation is also affine transformation. The method to solve this problems is introducing `Non-linear Transformation`. Using `Non-linear Function` to handle and transform there data, which is also called as `Activation Function` widely.

# #### ReLU Function
# `ReLU Function` provides a extremely simple `Non-linear Transformation`. For given element $x$, ReLU function is defined as  
# $$ReLU(x) = \max(x, 0)$$
# It is easy to understand that ReLU function just **retains positive elements, removes negative element**. 

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import d2lzh as d2l
from mxnet import autograd, nd

def xyploy(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')


# In[6]:


x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyploy(x, y, 'relu')


# $$relu'(x) = \begin{cases} 0 & for\ x \lt 0 \\ 1 & for\ x \ge 0\end{cases}$$

# In[7]:


y.backward()
xyploy(x, x.grad, 'grad of relu')


# #### Sigmoid Function
# `Sigmoid Function` can **transfrom elements between 0 and 1.**
# $$sigmoid(x) = \frac1{1 + \exp(-x)}$$

# In[8]:


with autograd.record():
    y = x.sigmoid()
xyploy(x, y, 'sigmoid')


# $$sigmoid'(x) = sigmoid(x)(1 - sigmoid(x))$$

# In[9]:


y.backward()
xyploy(x, x.grad, 'grad of sigmoid')


# #### Tanh Function
# `Tanh Function` can **transform elements between -1 and 1**.
# $$\tanh(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}$$

# In[10]:


with autograd.record():
    y = x.tanh()
xyploy(x, y, 'tanh')


# $$\tanh'(x) = 1 - \tanh^2(x)$$

# In[11]:


y.backward()
xyploy(x, x.grad, 'grad of tanh')


# ### Multilayer Perceptron
# $$H = \phi(XW_h + b_h)$$
# $$O = HW_o + b_o$$
# Among them, $\phi$ is activation function, such as relu, sigmoid, tanh and so on.

# ### Exercise
# Consult data to Learn about other activation function.

# > Answer: See Below.

# ### Other Activation Function

# #### Binary Step
# $$f(x) = \begin{cases} 0 & for x \lt 0 \\ 1 & for x \ge 0\end{cases}$$
# <br>
# $$f'(x) = \begin{cases} 0 & for\ x \neq 0 \\ ? & for\ x = 0\end{cases}$$

# #### ElliotSig Softsign
# $$f(x) = \frac{x}{1 + |x|}$$
# <br>
# $$f'(x) = \frac{x}{(1 + |x|)^2}$$

# #### ISRU
# $$f(x) = \frac{x}{\sqrt{1 + \alpha x^2}}$$
# <br>
# $$f'(x) = \left(\frac1{\sqrt{1 + \alpha x^2}}\right)^3$$

# #### ISRLU
# $$f(x) = \begin{cases} \frac{x}{\sqrt{1 + \alpha x^2}} & for\ x \lt 0 \\ x & for\ x \ge 0\end{cases}$$
# <br>
# $$f'(x) = \begin{cases} \left(\frac1{\sqrt{1 + \alpha x^2}}\right)^3 & for\ x \lt 0 \\ 1 & for\ x \ge 0 \end{cases}$$

# #### BReLU
# $$f(x) = \begin{cases} ReLU(x_i) & for\ i mod 2 = 0 \\ -ReLU(-x_i) & for\ i mod 2 \neq 0 \end{cases}$$
# <br>
# $$f'(x) = \begin{cases} ReLU'(x_i) & for\ i mod 2 = 0 \\ ReLU'(-x_i) & for\ i mod 2 \neq 0 \end{cases}$$

# #### PReLu
# $$f(\alpha, x) = \begin{cases} \alpha x & for\ x \lt 0 \\ x & for\ x \ge 0 \end{cases}$$
# <br>
# $$f'(\alpha, x) = \begin{cases} \alpha & for\ x \lt 0 \\ 1 & for\ x \ge 0 \end{cases}$$

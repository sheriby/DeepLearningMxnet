#!/usr/bin/env python
# coding: utf-8

# # Forward Propagation, Back Propagation and Computational Propagation

# ## Forward Propagation
# `Forward Propagation` means that calculate variable **from input layer to output layer** in neural network. Simply put, assume the input featrue is $x \in \mathbb R^d$ with out bias, then the intermediate variable as follows.
# $$z = W^{(1)}x$$
# Among which, $W \in R{h \times d}$ is the weight parameters, and intermediate variable $z \in \mathbb R^h$. Put $z$ into the activation function $\phi$, we can get that.
# $$h = \phi(z)$$
# $h$ is a intermediate variable as well, therefore we can know
# $$o = W^{(2)}h$$
# Assume the loss function is ${\scr l}$, the loss item is 
# $$L = {\scr l}(o, y)$$
# According to the definition of L2 norm regularization, given hyperparameter $\lambda$, the regularization item is
# $$s = \frac\lambda2(\parallel\!W^{(1)}\!\parallel_F^2 + \parallel\!W^{(2)}\!\parallel_F^2)$$
# Therefore, given data sample, the loss with regularization is 
# $$J = L + s$$
# Among which $J$ is **Objective Function**.

# ## Computational graph of Forward Propagation
# Here is a computational graph of forward propagation.  
# **Input variables** are $a$ and $b$. **Output variables** is $e$. **Intermediate variables** are $c$ and $d$.
# ![](https://mlln.cn/2018/07/02/tensorflow%E6%95%99%E7%A8%8B02-%E8%AE%A1%E7%AE%97%E5%9B%BE%E5%8F%8A%E5%85%B6%E5%AE%9E%E8%B7%B5/images/computational-graph.png)

# ## Back Propagation
# `Back Propagation` is the method to **calculate the gradient of parameters** in neural network. Generally speaking, back propagation follows the chain rule, from the output layer to input layer.  
# For input or output $X, Y, Z$ is tensor of arbitrary shape function $Y = f(X)$ and $Z = g(y)$. According to the chain of rule, we can get  
# $$\frac{\partial z}{\partial x} = prod\left(\frac{\partial z}{\partial y}, \frac{\partial y}{\partial x}\right)$$
# Among which $prod$ is matrix multiplication by some necessary procedure such as reverse and transpose.  
# For objective function $J = L + s$.
# $$\frac{\partial J}{\partial L} = 1,\quad \frac{\partial J}{\partial s} = 1$$
# Calculate the gradient of output layer.
# $$\frac{\partial J}{\partial o} = prod\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial o}\right) = \frac{\partial L}{\partial o}$$
# Calculate the gradient of two regularization parameters.
# $$\frac{\partial s}{\partial W^{(1)}} = \lambda W^{(1)},\quad\frac{\partial s}{\partial W^{(2)}} = \lambda W^{(2)}$$
# Now, we can get the gradient of $W_2$ which is the closest to output layer.
# $$\frac{\partial J}{\partial W^{(2)}} = prod\left(\frac{\partial J}{\partial o}, \frac{\partial o}{\partial W^{(2)}}\right) + prod\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial W^{(2)}}\right) = \frac{\partial J}{\partial o}h^{\mathsf T} + \lambda W^{(2)}$$
# Continue propagation along the hidden layer, we can get.
# $$\frac{\partial J}{\partial h} = prod\left(\frac{\partial J}{\partial o}, \frac{\partial o}{\partial h}\right) = W^{(2)\mathsf T}\frac{\partial J}{\partial o}$$
# The gradient of intermediate variable $z$ as follows.
# $$\frac{\partial J}{\partial z} = prob\left(\frac{\partial J}{\partial h}, \frac{\partial h}{\partial z}\right) = \frac{\partial J}{\partial h}\odot\phi'(z)$$
# Activation function $\phi$ is operated by element, so here using $\odot$ operation.  
# Finally, we can get the gradient of $W^{(i)}$ which is closest to input layer.
# $$\frac{\partial J}{\partial W^{(i)}} = prob\left(\frac{\partial J}{\partial z},\frac{\partial z}{\partial W^{(1)}}\right) + prob\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial W^{(1)}}\right) = \frac{\partial J}{\partial z}x^{\mathsf T} + \lambda W^{(1)}$$

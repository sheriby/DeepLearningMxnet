#!/usr/bin/env python
# coding: utf-8

# ### Softmax Regression
# `Linear Regression` is suitable when the output values are continous. In another condition, the output of the model can be discrete values like image category. For such discrete prediction problems, we can use `Classification Model` like `Softmax Regression`. Unlike `Linear Regression`, there are more than one output value in `Softmax Regression,` and introduce the softmax operation to make the output more suitale for prediction and training for discrete values.

# #### Classification Problem
# Considering a simple image catagory classification problem, the length and width of input images are 2 pixels and color is grayscala. We can tag them as $x1, x2, x3, x4$. Suppose that the true tag of these images are dog, cat or chicken, which correspond to discrete values $y1, y2, y3$.  
# Usually we use discrete numbers respresenting to different categroies, such as $y1 = 1,\ y2 = 2,\ y3 = 3$.the tag of a image is one of these discrete numbers.

# #### Softmax Regression Model
# In the above example, there are 4 features and 3 catagories, so there are 12 weight and 3 biases in this model. Every input will get 3 output $o_1,\ o_2, o_3$:  
# $$o_1 = x_1w_{11} + x_2w_{12} + x_3w_{13} + x_4w_{14} + b_1$$
# $$o_1 = x_1w_{21} + x_2w_{22} + x_3w_{23} + x_4w_{24} + b_2$$
# $$o_1 = x_1w_{31} + x_2w_{32} + x_3w_{33} + x_4w_{34} + b_3$$
# Like linear regression, softmax regression is also single layer neural network, as $o_1, o_2, o_3$ is directly depending on $x_1, x_2, x_3, x_4$.(It is Dense layer/Fully connected layer)

# #### Softmax Operation
# There are two problems of classification. First of all, the range of output values are uncertain. Therefore, It is difficult for us to judge the meaning of output value. Besides, due to the truly tag is discrete value, so it is hard for us to measure error.  
# Softmax Operation has solved these two problems above by using the following formula to convert ouput values to positive and sum of them is 1.
# $$\hat y_1, \hat y_2, \hat y_3 = softmax(o_1, o_2, o_3)$$
# and
# $$\hat y_1 = \frac{e^{o_1}}{\sum_{i=1}^3e^{o_i}}\ \hat y_2 = \frac{e^{o_2}}{\sum_{i=1}^3e^{o_i}}\ \hat  y_3 = \frac{e^{o_3}}{\sum_{i=1}^3e^{o_i}}$$
# and it is easy to see that $0 \le \hat y_1, \hat y_2, \hat y_3 \le 1 $ and $\hat y_1 + \hat y_2 + \hat y_3 = 1$
# so $$\mathop{\arg\max}\limits_{i}o_i = \mathop{\arg\max}\limits_{i}\hat y_i$$

# #### Single-Simple Classification Vector Calculation Expression
# $$W = \begin{bmatrix}w_{11} & w_{12} & w_{13} \\ w_{21} & w_{22} & w_{23} \\w_{31} & w_{32} & w_{33} \\w_{41} & w_{42} & w_{43} \end{bmatrix},\ b = \begin{bmatrix} b_1 & b_2 & b_3\end{bmatrix}$$  
# 
# The feature of simple $i$ is
# $$x^{(i)} = \begin{bmatrix}x_1^{i} & x_2^{2} & x_3^{3} & x_4^{4}\end{bmatrix}$$  
# 
# The output of the output layer is 
# $$o^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix}$$  
# 
# The predicted probability of dog, cat or chicken is 
# $$\hat y^{(i)} = \begin{bmatrix}\hat y_1^{(i)} & \hat y_2^{(i)} & \hat y_3^{(i)}\end{bmatrix}$$
# 
# The vector calculation expression of sample i by sofemax regression is following
# $$o^{(i)} = x^{(i)}W + b$$
# $$\hat y^{(i)} = softmax(o^{(i)})$$

# #### Minni-Batch Sample Classification Vector Calculation Expression
# Suppose that the batch size is $n$, the number of feature is $d$, the number of output catagories is $q$.The batch feature $X \in \mathbb{R}^{n \times d}$. The weight and biasof softmax regression are $W \in \mathbb{R}^{d \times q}$ and $b \in \mathbb{R}^{1 \times q}$.  
# So, the vector calculation expressions are following  
# $$O = XW + b$$
# $$\hat Y = softmax(O)$$

# #### Cross Entropy Loss Function
# `Square Loss Function` doesn't work well in classification model. We'd better find out a suitable loss function such as `Cross Entropy Loss Function` which is a common method to measure loss.  
# Here is corss entropy.
# $$H(y^{(i)},\ \hat y^{(i)}) = -\sum_{j=1}^qy_j^{(i)}\log\hat y_j^{(i)}$$
# $y_j^{(i)}$ is 0/1 element from $y^{(i)}$. Corss entropy only cares about predicted probability of correct catagory, because incorrect catagory $y^{(i)}$ is 0. we find that $y_k^{(i)} = 1$, so others is $0$. At this point, we can get that 
# $$H(y^{(i)},\ \hat y^{(i)}) = -\log \hat y_k^{(i)}$$
# Cross Entropy Loss Function is defined as below.
# $${\scr l}(\Theta) = \frac1n\sum_{i=1}^nH(y^{(i)},\ \hat y^{(i)})$$

# #### Model Prediction and Evaluation
# We will use accuracy ,which equals to correct predicted number divided by total, to evaluate the model.

# ### Exercise
# Consult the data to understand the similiarities between `Maximum Likelihood Estimation` and `Minimum Corss-Entropy Loss Function`.

# > Answer: We should understand what is maximum likelihood estimation first.

# #### Maximun Likelihood Estimation
# From a statistical standpoint, a given set of observations are random sample of unknowed populations. The goal of maximun likelihood estimation is to **make inferences about the population that is most likely generated the sample**.  
# Associated with each probability distribution is a unique parameter vector $\theta = [\theta_1, \theta_2, \theta_3, ..., \theta_k]^{\mathsf T}$. Evaluating the joint denstiy at the observed data sample $y = [y_1, y_2, y_3, ..., y_n]$ gives a real-values function.
# $$L_n(\theta) = L(\theta;y) = f_n(y;\theta)$$
# which is called likelihood function.
# The goal of maximun likelihood estimation is to find a the value of model parameters that maximum the likelihood function over the parameter space. That is  
# $$\hat\theta = \mathop{\arg\max}\limits_{\theta \in \Theta} \hat L_n(\theta; y)$$
# ##### Guassian Distribution
# $$f_{\mu, \Sigma} = \frac1{(2\pi)^{D/2}}\frac1{|\Sigma|^{1/2}}\exp\left\{{-\frac12(x - \mu)^{\mathsf T}}\Sigma^{-1}(x - \mu)\right\}$$
# The shape of the function determines by **mean $\mu$** and **covariance matrix $\Sigma$**.

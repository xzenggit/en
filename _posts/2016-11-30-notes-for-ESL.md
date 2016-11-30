---
layout: post
title: Notes for the Elements of Statistical Learning
tags: DataScience Statistics MachineLearning
---

# Notes for [the Elements of Statistical Learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576/ref=sr_1_1?ie=UTF8&qid=1480517413&sr=8-1&keywords=elements+of+statistical+learning)

## Ch2. Overview of supervised learning

Two simple approaches to prediction: least sqaures and nearest neighbors. The linear model makes huge assumptions about structure and yields stable but possibly inaccurate predicitons. The k-nearest neighbors makes very mild structural assumptions: its predicitons are often accurate but can be unstable. 


A large subset of the most popular techinques in use today are variants of these two simple procedures.

* Kernel methods use weights that decrease smoothly to zero with distance from the target point, rather than the effective 0/1 weights used by k-nearest neighbors.
* In high-dimensional spaces the distance kernels are modified to emphasize some variable more than others. 
* Local regression fits linear models by locally weighted least squares, rather than fitting constants locally.
* Linear models fit to a basis expansion of the original inputs allow arbitrarily complex models. 
* Projection pursuit and neural network models consist of summs of nonlinearly transformed linear models. 





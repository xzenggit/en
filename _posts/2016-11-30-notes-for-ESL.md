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

The errors of model fitting could be decomposed into two parts: variance and squared bias. Therefore, we will have this kind of bias-variance tradeoff.

More generally, as the model complexity increases, the variance tends to increase and the bias tends to decrease, and verse versa. 

## Ch3. Linear Methods for Regression

To test the hypothesis that a particular coefficient is zero, we can calculate the z-score. A large (absolute) value of z-score will lead to rejection of this hypothesis. The z-score measures the effect of dropping certain variable. 

If we need to test for the significant of groups of coefficients simultaneously. We can use the F statistic.

The Gauss-Markov theorem implies that the least sqaures estimator has the smallest mean sqaured error of all linear estimators with no bias. 

Best-subset selection finds for each k the subset of size k that gives smallest residual sum of squares. 

Rather than search through all possible subsets, we can seek a good path through them. Forward-stepwise selection starts with the intercept, and then sequentially adds into the model the predictor that most improves the fit. It is a greedy algorithm. 

Backward-stepwise selection starts with the full model, and sequentially deletes the predictor that has the least impact on the fit. The candidate for dropping is the variable with the smallest Z-score.

Shrinkage Methods: Ridge (L2 norm), Lasso (L1 norm), Elastic Net (combine both). Ridge regression may be preferred because it shrinks smoothly, rather than in discrete steps. Lasso falls somewhere between ridge regression and best subset regression, and enjoys some of the properties of each.

## Ch4. Linear Methods for Classification

Linear Discriminant Analysis (LDA) approaches the classification problems by assuming that the conditional probability desnity functions are both normally distributed with mean and covariance parameters, respectively. Under this assumption, the Bayes optimial solution is to predict points as being from the 2nd class if the log of the likelihood ratios is below some threshold T. 


Logistic regression does not assume any specific shapes of densities in the space of predictor variables, but LDA does. Logistic regression is based on maximum likelihood estimation. LDA is based on least squares estimation. 

It is generally felt that logistic regression is a safer, more robust bet than the LDA model, relying on fewer assumptions. 







---
layout: post
title: Notes for Artificial Intelligence
tags: AI Udacity
---


## Notes for [Artificial Intelligence for Robotics](https://www.udacity.com/course/artificial-intelligence-for-robotics--cs373)

### 1. Introduction

* Belief = probability
* Sense = product followed by normalization
* Move = convolution (addition)

localization: sense-move cycle.

Entropy = $-\sum_i p(x_i) log(p(x_i))$

Total probability: $P(A) = \sum_iP(A\vert B_i) P(B_i)$, which is also called convolution.

Bayes Rule: $P(A\vert B) = \frac{P(B\vert A) P(A)}{P(B)}$

### 2. Kalman Filters

* Kalman filter: continous, uni-modal
* Monte Carlo localization: discrete, multi-modal
* Partical filter: continous, multi-modal

 1-D Gaussian: $f(x) = \frac{1}{\sqrt{2\pi{\sigma}^2}}e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}$

* Performing a measurement means updating our belief by a multiplicative factor, while moving involves performing a convolution.
* Measurement meant updating our belief (and renormalizing our distribution). Motion meant keeping track of where all of our probability "went" when we moved (which meant using the law of Total Probability).

* Measurements: product, Bayes Rule
* Move: convolution, Total Probability

In Kalman Filter:

* Measurment update: product, Bayes rule
* Motion update (Prediction): convolution (addition), Total Probability
* Gaussian is used to combine them together

Mutiply two Gaussians with $\mu_1, \sigma_1^2$ and $\mu_2, \sigma_2^2$, we have the new Guassian with

* $\mu = \frac{\sigma_2^2 \mu_1 + \sigma_1^2 \mu_2}{\sigma_2^2+\sigma_1^2}$
* $\sigma^2 = \frac{1}{\frac{1}{\sigma_2^2}+\frac{1}{\sigma_1^2}}$
* This is the measurement update.

Motion update: $\mu_{new}$ <- $\mu + \nu$; $\sigma_{new}^2$ <- $\sigma^2+ \gamma^2$.

Kalman Filter can map states from "Observable" to "Hidden".

Suppose x is estimate, P is uncertainty covariance, F is state transition matrix, u is motion vector, Z is measurement, H is measurement function, R is measurement noise, I is identity matrix, then we have

* prediction: $x' = Fx+u$; $P'=FPF^T$
* measuremnt update: $y=Z-Hx$; $S=HPH^T+R$; $K=PH^TS^{-1}$; $x=x+KY$; $P' =(I-KH)P$.

### 3. Particle Filters

|  |state space | belief | efficiency | in robotics|
| -|:----------:| -------:|-------:|---------:|
|Histogram Filter| discrete| multimodal| exponential|approximate|
|Kalman Filter| continous| unimodal| quadratic | approximate|
|Particle Filter| continuous| multimodal|?|approximate|

Key advantageof Particle Filter is easy to program.

Resampling: Resample the particles with replacement and probability proportional to the importance weights.

* Measurement updates: $P(x\vert z) \propto P(z\vert x) P(x)$
* Motion updates: $P(x') = \sum P(x\vert x) P(x)$
* $P(x)$ is particles; $P(z\vert x)$ is importance weights; $P(x'\vert x)$ is sample

### 4. Search

Planning problem: Given map, starting location, goal location, and cost, find minimum cost path.

A-search: heuristic function is the optiaml guess of how far from the goal (withou obsticles)

Dynamical Progamming

### 5. PID Control

Smoothing algorithms: minimize $(x_i - y_i)^2 + \alpha (y_i - y_{i+1})^2$ using gradient descent.

* P controller:  steering = $-\tau_P * CTE$, CTE means cross track error.
* PD controller: steering = $-\tau_P * CTE - \tau_D * \frac{dCTE}{dt}$
* PID controller: steering = $-\tau_P * CTE - \tau_D * \frac{dCTE}{dt} - \tau_I * \sum CTE$
* P is proportional, to minimize error; I is integral, to compliment drift; D is differential, to avoid overshoot.

Twiddle (coordiane descent): run() -> goodness

```pyton
# twiddle algorithm
def twiddle(tol = 0.2):
#Make this tolerance bigger if you are timing out!
############## ADD CODE BELOW ####################
    n_params = 3
    dparams = [1.0 for row in range(n_params)]
    params = [0.0 for row in range(n_params)]

    best_error = run(params)
    n = 0
    while sum(dparams) > tol:
        for i in range(len(params)):
            params[i] += dparams[i]
            err = run(params)
            if err < best_error:
                best_error = err
                dparams[i] *= 1.1
            else:
                params[i] -= 2.0 * dparams[i]
                err = run(params)
                if err < best_error:
                    best_error = err
                    dparams[i] *= 1.1
                else:
                    params[i] += dparams[i]
                    dparams[i] *= 0.9
        n += 1
        print 'Twiddle #', n, params, '->', best_error
    print ' '
    return run(params)
```

### 6. SLAM

Simultaneous localizations and mapping (SLAM)







---
layout: post
title: Notes for Machine Learning at Udacity
tags: MachineLearning Udacity
---

## Notes for [Machine Learning at Udacity](https://www.udacity.com/course/machine-learning--ud262)

### 1. Decision Trees

Supervised Learning:

* Classification: output discrete
* Regression: output continuous

Classification learning:

* Instances (Input)
* Concepts (Functions)
* Targets
* Hypothesis 
* Sample (training set)
* Candidate 
* Testing set 

Decision Trees Learning:

* Pick best attribute
* Ask questions
* Follow the answer path
* Repeat the process unitl get the answer

[ID3](http://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart): find for each node the categorical feature that will yield the largest information gain for categorical targets.

ID3 algorithm:

* A <- best attribute
* Assign A as decision attribute for node
* Fore each value of A, create a descendant of node
* Sort training examples to leaves
* If example perfectly classified, STOP; else iterate over leaves.
* Loop the whole process until STOP.
* Prone if overfitting

Gain(S, A) = Entropy(S) - $\sum_r \frac{\vert S_r \vert}{\vert S\vert}Entropy(S_r)$

Entropy = $-\sum_rP(r)log(P(r))$

ID3 Bias: restrictive bias (hypothesis), preference bias(inductive bias).

### 2. Regression and classification

* [Generailzied regression](http://scikit-learn.org/stable/modules/linear_model.html)
* [Cross validaiton](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Model valuation](http://scikit-learn.org/stable/modules/model_evaluation.html)

### 3. Neural Network

Given example, find weights that map inputs to outputs

* perception rule (threshold)
* gradient decent rule (unthresholded)

Perception rule (single unit):

$ w_i = w_i +\Delta w_i$

$ \Delta w_i = \eta (y-\hat{y}) x_i $

$ \hat{y} = (\sum w_i x_i \le 0) $

$y$ is target; x is input; $\hat{y}$ is output $\eta$ is learning rate.

Gradient descent:

$ \hat{y} = a = \sum x_i w_i $

$ E(w) = \frac{1}{2} \sum (y-a)^2$

$ \frac{\partial E}{\partial w_i} = \sum(y-a)(-x_i) $

Comparison of learning rules:

* perception: guarantee finite convergences linear seperatability
* gradient descent: calculus robust, converge local optimal

Sigmoid function $ \sigma(a) = \frac{1}{1+e^{-a}}$, is the differentiable threshold.

So the whole structure of neural networks is differentiable.

Backpropogation:

* Computationally beneficial organizaiton of the chain rule
* Many local optimal

Optimizing weights:

* gradient descent
* momentum method
* higher order derivatives
* random optimization
* penalty for "complexity"

Restriction bias:

* representation power
* set of hypothesis we will consider
* perception: half spaces
* sigmoid: much more complex, not much restriction

Neual networks can represent:

* boolean functions: network of threshold-like units
* continous functions: "connected" no jumps - 1 hidden layer
* artibitrary functions: stitch together - two hidden layers

We need to use methods like cross-validation to avoid overfitting.

Preference bias: algorithms selection of one representation over another

Initiall weights: small random values, small means low complexity, random may avoid local minimums.

DO NOT make more complex network unless you can get smaller error.

### 4. Instance based learning

Put all data into a database, do a look up.

* remember extactly
* fast and simple
* no generalizations

K-NN(K nearest neighor):

Given: training data D = {$x_i, y_i$}; distance metric d(q, x), number of neighbors k, query point q.

NN = {i: d(q, $x_i$) k smallest}

Return:

* classification: weighted vote of the $y_i \in$ NN
* regression: weighted mean of the $y_i \in$ NN

K-NN bias:

* perference bias: our belief about what makes a good hypothesis
	* locality: near points that are similar
	* smoothness: averaging
	* all featrues matter equally

Curse of dimensionality: 

As the number of features or dimensions grow, the amount of data we need to generalize accurately grows exponentially!

For machine learning, that means more data and less features.

### 5. Ensemble learning: boosting

* Learn over a subset of data -> get rules
* Combine them together to get the generalized rule

Ensemble methods:

* Bootstrap:
	* Given a dataset of size N;
	* Draw N samples with replacement to create a new dataset;
	* Repeat ~1000 times and get ~1000 sample datasets;
	* Compute ~1000 sample statistics.

* Bagging:
	* Draw N bootstrap samples;
	* Retrain the model on each sample;
	* Average the results (for regression, do averaging; for classification, do majority vote);
	* Works great for overfit models (decrease variance without changing bias; doesn’t help much with underfit/high bias model)

* Boosting:
	* Instead of selecting data randomly with the bootstrap, favor the misclassified points
	* Initialize the weights
	* Repeat: resample with respect to weights; retrain the model; recompute weights

Ensemble method combines simple models with bad resutls together to get better results. To me, it's like the diverse investiment theory in finance. By diversifying your investiment, you can get lower risk. 

Boosting error estimate: instead of using number of mismatches, the probability of each mismatche $P[h(x) \ne c(x)]$ will be considered.

"Weak" learner: does better than change always. For all data distribution, error rate is less than half.

Boosting for binary classificatin (AdaBoost):

* Given traning examples $(x_i, y_i)$, $y_i$ in {-1, 1}
* Initialize $D_1(i) = 1/m$.
* For t = 1 to T
	* Train weaker learner using distribution $D_t$
	* Get weak classifier $h_t$
	* Choose $\alpha_t$
	* Update $D_{t+1}(i) = \frac{D_t(i) exp{-\alpha_t y_i h_t(x_i)}}{Z_t}$, where $\alpha_t = \frac{1}{2}ln(\frac{1-\epsilon_t}{\epsilon_t})$, $Z_t$ is normalization factor $Z_t = \sum_i D_t(i) exp(-\alpha_t y_i h_t(x_i))$.
* Output final classifier: $H(x) = sign(\sum_{t=1}{T} \alpha_t h_t(x))$.

The weak learner's job is to find a weak classifier $h_t$ for the distribution $D_t$. In the binary case, it is to minimize the error $\epsilon_t = P[h_t(x_i) \ne y_i]$.

For boosting, it's much harder to get overfitting. 

A nice overview of boosting method:[The Boosting Approach to Machine Learning An Overview](https://www.cs.princeton.edu/courses/archive/spring07/cos424/papers/boosting-survey.pdf).

Here are some notes from that paper.

Boosting is based on the observation that finding many rough rules of thumb can be a lot of easier than finding a single, highly accurate prediction rule. To apply the boosting approach, we start with a method for finding the rough rules of thumb, which is called "weak" or "base" learner, repeatedly. Each time we feed it a different subset of the training exammples with different distribution or weighting. The "weak" learner generates a new weak prediction rule. After many rounds, the boosting mehtod must combine these weak rules into a single prediction rule that will be much more accurate than any one of the weak rules. 

Like I mentioned before, the principle is similar as the diversifying investiment strategy, by doing that you can get lower risk than each strategy does.

To make the boosting approache work, two questions must be answered: 

* how should each distribution be chosen on each round?
* how should the weak rules be combined into a single rule?

Regarding the choice of distribution, the method is to place most weight on the examples most offten misclassified by the preceding weak rules, which forces the weak learner to focus on the "hardest" examples. For combing weak rules, a natural and effective way is taking a (weighted) majority vote for their prediction.

Boosting method may avoid overfitting, because it's like the SVMs by increasing the margins. It tends to overifft if weak learner uses ANN with many layers and nodes.

* Pink noise: uniform noise
* White noise: Guassian noise

### 6. Kernel methods and Support Vector Machines

$y = w^T x + b$ , y is classification label, w and brepresenet parameters of the plane.

$ w^T x+ b = 0$ is the decision boundary. 

$ w^T x + b = \pm 1$ are the margin.

$ \frac{w^T}{\|w\|}(x_1 - x_2 = \frac{2}{\|w\|})$, the l.h.s is the margin we need to maximize while classifying everything correctly $y_i (w^T + b) \ge 1, \forall i$.

That is equal to minimize $1/2 \|w\|^2$, which makes optimization problem easier.

It turns out it also equals to maximize $W(\alpha) = \sum_i \alpha_i - 1/2 \sum_{i.j} \alpha_i \alpha_j y_i x_i^T x_j, s.t., \alpha_i \ge 0, \sum_i \alpha_i y_i = 0$. So we can get $w = \sum \alpha_i y_i x_i$ and b. It turns out that $\alpha_i$ mostly are zeros. That means only a few points matter. 

We can use kernel functions to represent the similarity, and solve the problem in a different dimension. 

### 7. Computational learning theory

Computatioanl learning theory:

* define learning problems
* show specific algorithms work
* show thesse problems are fundamentally hard

Theory of computing analyzes how algorithms use resourse: time, space et .

Defining inductive learning - learning from examples:

* probability of successful training
* number of examples to train on
* complexity of hypothesis class
* accuracy to which target concept is appropriate
* manner in whihch training examples presented (batch, online)
* manner in which training examples selected

Select training examples:

* Learner asks questions of teacher
* Teacher gives examples to help learner
* Fixed distribution

Teacher with constrained queries:

* X: x1, x2, ..., xk; k-bit input
* H: conjunctions of literals or negation

Computationl complexity: how much computational effort is needed for a learner to change?

Sample comoplexity - batch: how many training examples are needed for a learner to create successful hypothesis?

Mistake bounds - online: how many misclassifications can a learner make over an infinite run?

PAC learning - error of hypothesis:

* Training error: fraction of training examples misclassified by h
* True error: fraction of examples that would be misclassified on sample draw from D

$error_D(h) = P_{x~D}[c(x) \ne h(x)]$

Notations: 

* C: concept class
* L: learner
* H: hypothesis space
* n: size of hypothesis space
* D: distribution over inputs
* $0\le \epsilon \le 1/2 $: error goal
* $0 \ sigma \le 1/2 $: certainty goal ($1-\sigma$)

C is PAC-learnable by L using H if learner L will, with probability $1-\epsilon$, output a hypothesis $h \in H$ s.t. $error_D(h) \le \epsilon$ in time and spae and samples polynomial in $1/\epsilon, 1/\sigma$ and n.

VS(s) is the set of trainning hypotheses in H that perfectly fit the training data.

VS(s) is $\epsilon$-exhuasted iff $\forall h \in$ VS(s), $error_D(h) \le \epsilon$.

Haussler Theorem - Bound True Error: Let $error_D(h_1,..., h_k \in H) > \epsilon$ be the high true error, how much data do we need to "knock off"  these hypothesis?

$ P(h_i(x) = c(x)) \le 1-\epsilon$, if everything is independent, we have P($h_i$ consistent with c on m exmaples) $\le (1-\epsilon)^m$. P(at least one of $h_i$ consistent with c on m exmaples) $\le k(1-\epsilon)^m \le \vert H\vert (1-\epsilon)^m \le \vert H\vert e^{-\epsilon m} \le \delta$. This is the upper bound that version space is not $\epsilon$-exhusted after m samples. That is, $m \le \frac{1}{\epsilon} (ln(\vert H\vert) + ln \frac{1}{\delta})$.

Because $-\epsilon \ge ln(1-\epsilon)$, $(1-\epsilon)^m \le e^{-\epsilon m}$.

If H is finite, and m is a sequence of independent randomly drawn training points, then for any $0 \le \epsilon \le 1$, the probability that the version space is not $\epsilon$-exhausted is bounded above by $\vert H\vert e^{-\epsilon m}$.

### 8. VC dimensions

Infinite hypothesis spaces: linear separators, ANN, continuous decision trees.

VC dimension: the size of the largest subset of input that the hypothesis class can shatter, it can be considered a measure of complexity of a hypothesis space.

For d-dimensional hyperplane, the VC dimensionis d+1.

Sample complexity and VC dimension: $m \ge \frac{1}{\epsilon} (8 VC(H) log_2^(\frac{13}{\epsilon})+4log_2^{\frac{2}{8}})$ for infinite case, $m \ge (ln(\vert H\vert) + ln(\frac{1}{8}))$ for finite case.

For finite case, $d \le log_2(\vert H\vert)$. 

H is PAC-learnable if and only if VC dimension is finite.

### 9. Bayesian learning

Bayesian learning: learn the most probable hypothesis given data and domain knowledge.

$P(h\vert D) = \frac{P(D\vert h)P(h)}{P(D)}$

$P(a,b) = P(a\vert b) P(b) = P(b\vert a) P(a)$

P(D\vert h) is data given the hypothesis, P(h) is the prior on data, P(D) is prior on data.

For each $h \in H$, calculate $P(h\vert D) = P(D \vert h) P(h) / P(D)$. 

Output (suppose P(h) is uniform):

* map -> maximum posterior: $h_m = argmax P(h\vert D)$
* machine learning -> maximum likelihood: $h_ML = argmax P(D\vert h)$. 

### 10. Bayesian inference

* Representing and resasoning with probabilities
* Bayesian Networks

X is conditionally independent of Y given Z if the probability distribution governing X is independent of the value of Y given the value of Z; that is P(X\vert Y,Z) = P(X\vert Z).

Belief Networks aka Bayesian Networks aka Graphical Models

Why sampling?

* get probability of values, and generate values for certain distribution
* simulation of a complex process
* approximate inference
* visualization

Inferencing rules:

* Marginalization: $P(x) = \sum_y P(x, y)$
* Chain rule: P(x, y) = P(x)P(y\vert x)
* Bayes rule: P(y\vert x) = P(x\vert y)P(y)/P(x)

### 11. Randomized optimization

Optimization: 

Input space X, objective function f(x), goal is to find $\hat{x} \in X$ s.t. $f(\hat{x})=max f(x)$.

Optimization approaches:

* Newton methods: single optimum problem wit derivatives
* Randomized optimization: big input space, complex function, no derivative or hard to find.

Hill climbing:

* Guess $x \in X$
* Repeat:
	* let $x^* = argmax f(x) for x \in N(x)$, N means neighborhood
	* if $f(x^*) > f(x)$: x = n
	* else: stop

Randomized hill climping: once local optimum is reached, try gain starting from a randomly chosen x. 

Advantages: multiple tries to find a good starting place, not much more expensive (constant factor).

Simulated Annealing: don't always improve, sometimes you need to search (explore). 

For a finite set of iterations:

* Sample new point $x_+$ in N(x)
* Jump to new sample with probability given by an acceptance probability function $P(x, x_+, T)
* Decrease temperate T

$$P(x, x_+, T) = \left \{ \begin{tabular}{cc} 1 & if f(x_+) \ge f(x)\\ e^{\frac{f(x_+)-f(x)}{T}} & otherwise \end{tabular}$$

Properties of Simulated Annealing: 

* T -> $\inf$ means likely to move freely (random wal), T -> 0 means go uphill. 
* P(ending at x) = $\frac{e^{f(x)/T}}{Z_T}$ Boltzman distribution

Genetic Algorithms (GA):

mutation - local search; cross over - population holds information; generations - iterationos of improvement. 

```
# GA skeleton
P0 = initial population of size K
repeat until converged
  compute fitness of all x 
  select "most fit" individuals (top half, weighted probability)
  pair up individuals, replacing "least fit" individuals via crossover/mutation
```

The above methods:

* only points, no structure
* unclear probability distribution

MIMIC method: convey structure, direct model distribution.

$$P^{\theta}(x) = \left \{\begin{tabular}{cc} \frac{1}{z_{\theta}} & if f(x) \ge \theta\\ 0 & otherwise  $$

$P^{\theta_{min}}(x) = uniform$; $$P^{\theta_{max}}(x) = optima$

MIMIC pseudo code:

Generate samples from $P^{\theta}(x)$; 
Set $\theta_{t+1}$ to nth percentice;
Retain only those samples s.t. $f(x) \gt \theta_{t+1}$;
Estimate $P^{\theta_{t+1}}(x)$;
Repeat.

### 12. Clustering

Basic clustering problem:

* Given set of object x, inter-object distance D(x,y) = D(y,x)
* Output: partition $P_D(x) = P_D(y)$ if x and y in same cluster

Single linkage clustering (SLC):

* consider each object a cluster (n object)
* define intercluster distance as the distance between the closest two points in the two clusters
* merge two closet clusters
* repeat n-k times to make n cluster

Runing time of SLC with n points and k clusters is $O(n^3)$.

K-means clustering:

* pick k centers at random
* each center claims its closest points
* recompute the centers by averaging the clustered points
* repeat until convergence

Properties of K-means clustering:

* each iteration polynomial O(kn)
* finite (exponential) iterations $O(k^n)$
* error decreases if ties broken consistently
* can get stuck (local optima)

Soft clustering:

* Assume the data was generated by 
	* select one of k Gaussians uniformly
	* sampel $x_i$ from that Gaussian
	* repeat n times
* Task: find a hypothesis that maximizes the probability of the data

The machine learning mean of the Gaussian mean is the mean of the data.

Expectation maximization:

$$ E[z_{ij}] = \frac{P(x=x_i|\mu=\mu_j)}{\sum_i^kP(x=x_i|\mu=\mu_j)}$$

$$ \mu_{j} = \frac{\sum_i E[z_{ij}]x_i}{\sum_iE[z_{ij}]}$$

Properties of EM:

* monotonically non-decreasing likelihood
* does not have to converge (practically does)
* will not diverge
* can get stuck
* works with any distribution (if EM solvable)

Clustering properties:

* Richness: for any assignment of objects to clusters, there is some distance matrix D s.t. $P_D$ returns that clustering $\forall c \in D$, $P_D=c$.
* Scale-invariance: scaling distance by a positive value doesn't change the clustering.
* Consistency: shrinking intracluster distances and expanding intercluster distance does not change the clustering.

Impossibility theorem: no clustering scheme can achieve all three of richness, scal invariance, and consistency.

### 13. Feature selection

* Knowledge discovery
* Curse of dimensionality

Two methods: 

* filtering: faster, isolated featrues, ignores the learning problem
	* information gain, variance, entropy
* wrapping: takes into account model bias, slow
	* hill climping, randomized optimization, forward, backward

Relevance

* $x_i$ is strongly relevant if removing it degrades Bayes Optimal Classifier(BOC)
* $x_i$ is weakly relevant if 
	* not strongly relevant
	* there's a subset of features s.t. adding $x_i$ to this subset improves BOC
* otherwise, $x_i$ is irrelevant

### 14. Feature transformation

The problem of pre-processing a set of features to create a new feature set, while retaining as much information as possible.

* PCA: correlation, maximize variance
* ICA: independence of new feature
* RCA: generate random directions
* LDA: linear discriminant analysis, find a projection that descriminates based on the label

### 15. Information theory

* Mutual information: are the input vectors similar?
* Entropy: does each feature have any information?

Entropy: $H(x) = -\sum P(x) logP(x)$

Joint Entropy: $H(x,y) = -\sum P(x, y) logP(x,y)$

Conditional Entropy: $H(y|x) = -\sum P(x, y) logP(y|x)$

Mutual Information: I(x,y) = H(y) - H(x|y)

KL divergence: measure distance between two distributions $D(p|q) = \int p(x) log(\frac{p(x)}{q(x)})$

### 16. Markov decision process

* State: s
* Model: $T(s,a,s') ~ P(s'|s, a)$
* Actions: A(s), A
* Reward: R(s), R(s,a), $R(s,a,s')$
* Policy: $\Pi(s)$ -> a, $\Pi^*$ is the optimal policy.

Sequence of rewards

* infinite horizons -> stationary
* Utility of sequeneces -> stationary preference, if $U(s_0, s_1, \ldots) > U(s_0, s_1^', \ldots)$, then $U(s_1, \ldots) > U(s_1^', \ldots)$.

* $U(s_0, s_1, s_2, \ldots) = \sum_{t=0}^{\inf} R(s_t)$. This cannot tell the difference between two sequences if they all go to infinity
* $U(s_0, s_1, s_2, \ldots) = \sum_{t=0}^{\inf} \gamma^t R(s_t) \le \sum_{t=0}^{\inf} \gamma^t R_{max} = \frac{R_{max}}{1-\gamma}$, $0 \le \gamma <1$. This is similar as the utility definition in economics.

Polices - Bellman Euqation: 

* $\Pi^* = \argmax_{\Pi} E[\sum_{t=0}^{\inf} \gamma^t R(s_t) | \Pi]$

* $ P^{\pi}(s) = E[\sum_{t=0}^{\inf} \gamma^t R(s_t)| \Pi, s_0 = s]$

* $\Pi^* = \argmax_{a} \sum_{s'} T(s,a,s') U(s')$

* $U(s) = R(s) + \gamma \max_a \sum_{s'} T(s, a, s') U(s')$

Reward is immediate, and utility is long-term reward.

Value iteration:

* start with arbitrary utility
* update utility based on neigbors
* repeat until converge

Policy iteration:

* start with $\pi_0$ by guessing
* evaluate: given $\Pi_t$ calculate $u_t^{\Pi}$
* improve $\Pi_{t+1}=\argmax_a \sum T(s,a,s') U_t(s')$
* $ U_t(s) = R(s) + \gamma \sum_{s'} T(s, \Pi_t(s), s') U_t(s')$

The Bellman Equation: 

* $V(s) = \max_a(R(s,a) + \gamma \sum_{s'}T(s,a,s')V(s'))$, V here is the value function. 
* $Q(s,a) = R(s,a) +\gamma \sum_{s'}T(s,a,s') \max_{a'} Q(s',a')$, Q is quanlity
* $C(s) = \gamma \sum_{s'} T(s,a,s') \max_{a'}(R(s',a')+C(s',a'))$, C is the continuation.

* $V(s) = \max_aQ(s,a)$
* $V(s) = \max_a(R(s,a) + C(s,a))$
* $Q(s,a) = R(s,a) + \gamma\sum_{s'}T(s,a,s') V(s')$
* $Q(s,a) = R(s,a) +C(s,a)$
* $C(s,a) = \gamma \sum_{s'}T(s,a,s')V(s')$
* $C(s,a) = \gamma \sum_{s'}T(s,a,s')\max_{a'}Q(s',a')$

### 17. Reinforcement learning

* Planning: Model(T, R) -> PLANNER -> Policy($\pi$)
* Reinforcement learning: Transitions(<s,a,r,s'>) -> LEARNER -> Policy

* Model-based RL: Transitions -> MODLER -> Model -> PLANNER -> Policy
* RL-based planner: Model -> SIMULATOR -> Transitions -> LEANDER -> Policy

Three approaches to RL:

* Policy search: direct use indirect learning s->$\Pi$->a
* Value-function based: s->U->v
* Model-based: direct learning indirec use <s,a> -> <T, R> -> <s', r> 

A new kind of value function:

* $ U(s) = R(s) + \gamma \max_q \sum_{s'} T(s, a, s') U(s')$
* $ \Pi(s) = \argmax_a \sum_{s'} T(s, a, s') U(s')$

Q-learning:

* $ Q(s, a) = R(s) + \gamma \sum_{s'} T(s, a, s') \max_{a'}Q(s',a')$, value for arrving in s, leaving via a, proceeding optimaally thereafter.
* $ U(s) = \max_a Q(s,a) $
* $ \Pi(s) = \argmax_a Q(s,a) $

$\hat{Q}(s,a) =  (1-\alpha_t) (R(s)+ \gamma \max_{a'}\hat{Q}(s',q'))+\alpha_t (R(s)+ \gamma \max_{a'}\hat{Q}(s',q'))$

Q-learning is a family of algorithms:

* how initialize $\hat{Q}$?
* how decay $\alpha_t$?
* how choose actions?

$\epsilon$-Greedy exploration: "greedy limit + infinite exploration", exploration-exploiation dilemma.

Temporal Difference Learning: learn to predict over time 

Properties of learning rates: $V_T(s) = V_{T-1}(s) + \alpha_T(R_T(s)-V_{T-1}(s))$; $\lim_{T->\inf}V_T(s) = V(s)$; $\sum_T \alpha_T=\inf, \sum_T\alpha_T^2 <\inf$.

TD(1) Rule:

* Episode T
	* For all s, e(s) = 0 at start of episode, $V_T(s) = V_{T-1}(s)
	* After $s_{T-1}$->$s_T$ with reward $r_t$: $e(s_{T-1})=e(s_{T-1})+1$
* For all s,
	* $V_T(s) = V_T(s) + \alpha_T (r_t + \gamma V_{T-1}(s_t)-V_{T-1}(s_{T-1}))e(s)$; $e(s) = \gamma e(s)$

$TD(\lambda)$ Rule:

* Episode T
	* For all s, e(s) = 0 at start of episode, $V_T(s) = V_{T-1}(s)
	* After $s_{T-1}$->$s_T$ with reward $r_t$: $e(s_{T-1})=e(s_{T-1})+1$
* For all s,
	* $V_T(s) = V_T(s) + \alpha_T (r_t + \gamma V_{T-1}(s_t)-V_{T-1}(s_{T-1}))e(s)$; $e(s) = \lambda \gamma e(s)$

Bellman operator: let B be an operator, or mapping from value functions to value functions $[BQ](s,a)=R(s,a) +\gamma \sum_{s'}T(s,a,s') \max_{a'}Q(s',a')$; $Q^*=BQ^*$ is Bellman Equation; $Q_T=BQ_{T-1}$ is Value Iteration.

Contraction Mappings: B is an operator, if for all F, G and some $0 \le \r <1$, $\|BF-BG\|_{\inf} \le r \|F-G\|_{\inf}$, then B is a contraction mapping.

Contraction Properties: if B is a contraction mapping,

* $F^* = BF^*$ has a solution and it is unique
* $F_t = BF_{t-1}$ => $F_t$ -> $F^*$ (value iteration converges)

Bellman operator contracts:

* $[BQ](s,a) = R(s,a) + \gamma \sum_{s'}T(s,a,s') \max_{a'}Q(s',a')$
* Given $Q_1, Q_2$, $\|BQ_1-BQ_2\|_{\inf}$ = \max_{a,s}|\gamma\sum_{s'}T(s,a,s')(\max_{a'}Q_1(s',a')-\max_{a'}Q_2(s',a'))| \le \gamma \max_{s'}|\max_{a'}Q_1(s',a')-\max_{a'}Q_2(s',a')| \le \gamma \max_{s',a'}|Q_1(s',a')-Q_2(s',a')|=\gamma \|Q_1-Q_2\|_{\inf}$

Why might we want to change the reward function for a MDP?

* Easier to solve/represent and similar to what it would have learned
* Because we don't have one yet

How can we change the MDP reward function without changing the optimal policy?

* multiply by positive constant
* shift by constant (add)
* nonlinear potential-based

### 18. Game theory

In a 2-player, zero-sum deterministic game of pefect information, minmax=maxmin and there always exists an optimal pure strategy for each player.

Nash Equilibrium: n players with strategies $s_1, s_2, \ldots, s_n$, $s_1^* \in s_1, s_2^* \in s_2, \ldots, s_n^* \in s_n$ are a Nash Equilibrium if and only if $\forall i, s_i^* = \argmax_{s_i} utility(s_1^*, \ldots, s_i, \ldots, s_n^*)$.

* In the n-player pure strategy game, if elimination of all strictly nominated strategies, eliminates all but one combination of strategies, then that combination is in fact the unique Nash equilibrium.
* Any Nash equilibrium will survive the iterated elimination of strictly dominated strategies. In other words if you get rid of things that are strictly dominated you will not accidentally get rid of nash equilibria in the process.
* If n is finite, that is you have a finite number of players, and for each of the set of strategies, that set of strategies is also finite, in other words you're still in a finite game, then there exists at least one Nash equilibrium which might involve mixed strategies.

In game theory, Folk Theorm refers to a particular result: describes the set of payoffs that can result from Nash strategies in repeated games.

Folk Theorem: any feasible payoff profile that strictly dominates the minmax/security lelve profile can be realized as a Nash equilibrium payoff profile with sufficiently large discount factor.

Stochastic Games: 

* S: states
* $A_i$: actions for player i
* T: transitions
* $R_i$: rewards for player i
* $\gamma$: discount




































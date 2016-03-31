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

Gain(S, A) = Entropy(S) - $\sum_r \frac{|S_r|}{|S|}Entropy(S_r) $

Entropy = $-\sum_rP(r)log(P(r))$

ID3 Bias: restrictive bias (hypothesis), preference bias(inductive bias).

### 2. Regression and classification

* [Generailzied regression](http://scikit-learn.org/stable/modules/linear_model.html)
* [Cross validaiton](http://scikit-learn.org/stable/modules/cross_validation.html)
* [Model valuation](http://scikit-learn.org/stable/modules/model_evaluation.html)

### 3. Neural Network








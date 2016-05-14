---
layout: post
title: Notes for common aglorithms
tags: aglorithm
---
**These notes are mainly copied from [geeksforgeeks](http://www.geeksforgeeks.org/greedy-algorithms-set-1-activity-selection-problem/).**

**A collection of algorithms can be found at [daqwest](www.daqwest.com).**

### [Common Algorithms](http://www.geeksforgeeks.org/fundamentals-of-algorithms/#DynamicProgramming)

* Brute Force: A brute-force algorithm solves a problem in the most simple, direct or obvious way. As a result, such an algorithm can end up doing far more work to solve a given problem than a more clever or sophisticated algorithm might do.  Typically, a brute-force algorithm solves a problem by exhaustively enumerating all the possibilities. 

* Divide and Conquer: works by recursively breaking down a problem into two or more sub-problems of the same (or related) type (divide), until these become simple enough to be solved directly (conquer). The solutions to the sub-problems are then combined to give a solution to the original problem.

* Greedy Programming: an algorithm that follows the problem solving heuristic of making the locally optimal choice at each stage[1] with the hope of finding a global optimum. In many problems, a greedy strategy does not in general produce an optimal solution, but nonetheless a greedy heuristic may yield locally optimal solutions that approximate a global optimal solution in a reasonable time.

* Dynamic Programming: a method for solving a complex problem by breaking it down into a collection of simpler subproblems, solving each of those subproblems just once, and storing their solutions - ideally, using a memory-based data structure. The next time the same subproblem occurs, instead of recomputing its solution, one simply looks up the previously computed solution, thereby saving computation time at the expense of a (hopefully) modest expenditure in storage space.

### [Complexity](http://bigocheatsheet.com/)

* Define complexity as a numerical function T(n) - time versus the input size n.
* Big-Oh (Upper Bound): $$f(n) = O(g(n)) \Leftrightarrow \exists c > 0, n_0 > 0$$, s.t. $$f(n) \le cg(n)$$, for all $$n\ge n_0$$.
* Big-Omega (Lower Bound): $$f(n) = \Omega (g(n)) \Leftrightarrow \exists c>0, n_0>0$$, s.t. $$f(n) \ge cg(n)$$, for all $$n\ge n_0$$.
* Big-theta: $$f(n)=\Theta(g(n)) \Leftrightarrow \exists f(n) = O(g(n))$$ and $$f(n)=\Omega(g(n))$$.
* Constant time: $$O(1)$$, run in constant time regardless of input size.
* Linear time: $$O(n)$$, run time propotional to input size n.
* Log time: $$O(log(n))$$, run time propotional to log of input size n.
* Quadratic time: $$O(n^2)$$, run time proportional to square of input size n.
* [Analysis of Algorithm](http://www.geeksforgeeks.org/analysis-of-algorithms-set-4-analysis-of-loops/)
* Master Method: T(n) = aT(n/b) + f(n), where a>=1 and b > 1.
    * if $$f(n)=\Theta(n^c)$$ where $$c<log_b^a$$, then $$T(n) = \Theta (n^{log_b^a})$$
    * if $$f(n) = \Theta(n^c)$$ where $$c=log_b^a$$, then $$T(n) = \Theta (n^c{log_b^a})$$
    * if $$f(n) = \Theta(n^c)$$ where $$c>log_b^a$$, then $$T(n) = \Theta (f(n))$$

![Big-O Complexity Chart](http://bigocheatsheet.com/img/big-o-complexity.png)

### Dynamic Programming Problems
Two main properties of a probelm that can be sovled using Dynamic Programming:

* Overlapping Subproblems
* Optimal Substructure

#### [Overlapping subproblems property](http://www.geeksforgeeks.org/dynamic-programming-set-1/)
 Dynamic Programming is mainly used when solutions of same subproblems are needed again and again. In dynamic programming, computed solutions to subproblems are stored in a table so that these don’t have to recomputed.

Two different ways to sotre the values so that these values can be reused:

* Memoization (Top Down)
* Tabulation (Bottom Up)

a) Memoization (Top Down)

The memoized program for a problem is similar to the recursive version with a small modification that it looks into a lookup table before computing solutions. We initialize a lookup array with all initial values as NULL. Whenever we need solution to a subproblem, we first look into the lookup table. If the precomputed value is there then we return that value, otherwise we calculate the value and put the result in lookup table so that it can be reused later.


```python
# Python program for Memoized version of nth Fibonacci number
 
# function to calcualte nth Fibonacci number
def fib(n, lookup):
 
    # Base case
    if n == 0 or n == 1 :
        lookup[n] = n
 
    # If the value is not calculated previously then calculate it
    if lookup[n] is None:
        lookup[n] = fib(n-1 , lookup)  + fib(n-2 , lookup) 
 
    # return the value corresponding to that value of n
    return lookup[n]
# end of function
# This code is contributed by Nikhil Kumar Singh
```

b) Tabulation (Bottom Up)

The tabulated program for a given problem builds a table in bottom up fashion and returns the last entry from table.


```python
# Python program Tabulated (bottom up) version
def fib(n):
 
    # array declaration
    f = [0]*(n+1)
 
    # base case assignment
    f[1] = 1
 
    # calculating the fibonacci and storing the values
    for i in xrange(2 , n+1):
        f[i] = f[i-1] + f[i-2]
    return f[n]
# This code is contributed by Nikhil Kumar Singh
```

In Memoized version, table is filled on demand while in tabulated version, starting from the first entry, all entries are filled one by one. Unlike the tabulated version, all entries of the lookup table are not necessarily filled in memoized version. For example, memoized solution of LCS problem doesn’t necessarily fill all entries.
 
#### [Optimal Substructure Property](http://www.geeksforgeeks.org/dynamic-programming-set-2-optimal-substructure-property/)

A given problems has Optimal Substructure Property if optimal solution of the given problem can be obtained by using optimal solutions of its subproblems.

For example the shortest path problem has following optimal substructure property: If a node x lies in the shortest path from a source node u to destination node v then the shortest path from u to v is combination of shortest path from u to x and shortest path from x to v.

On the other hand the Longest path problem doesn’t have the Optimal Substructure property. Here by Longest Path we mean longest simple path (path without cycle) between two nodes.

#### [Longest Increasing Subsequence](http://www.geeksforgeeks.org/dynamic-programming-set-3-longest-increasing-subsequence/)

The longest Increasing Subsequence (LIS) problem is to find the length of the longest subsequence of a given sequence such that all elements of the subsequence are sorted in increasing order. For example, length of LIS for { 10, 22, 9, 33, 21, 50, 41, 60, 80 } is 6 and LIS is {10, 22, 33, 50, 60, 80}.

Let arr[0..n-1] be the input array and L(i) be the length of the LIS till index i such that arr[i] is part of LIS and arr[i] is the last element in LIS, then L(i) can be recursively written as:

L(i) = { 1 + Max ( L(j) ) } 

where j < i and arr[j] < arr[i] and if there is no such j then L(i) = 1

To get LIS of a given array, we need to return max(L(i)) where 0 < i < n So the LIS problem has optimal substructure property as the main problem can be solved using solutions to subproblems.


```python
# Dynamic programming Python implementation of LIS problem
 
# lis returns length of the longest increasing subsequence
# in arr of size n
def lis(arr):
    n = len(arr)
 
    # Declare the list (array) for LIS and initialize LIS
    # values for all indexes
    lis = [1]*n
 
    # Compute optimized LIS values in bottom up manner
    for i in range (1 , n):
        for j in range(0 , i):
            if arr[i] > arr[j] and lis[i]< lis[j] + 1 :
                lis[i] = lis[j]+1
 
    # Initialize maximum to 0 to get the maximum of all
    # LIS
    maximum = 0
 
    # Pick maximum of all LIS values
    for i in range(n):
        maximum = max(maximum , lis[i])
 
    return maximum
# end of lis function
# This code is contributed by Nikhil Kumar Singh
```

#### [Longest Common Subsequence](http://www.geeksforgeeks.org/dynamic-programming-set-4-longest-common-subsequence/)

Given two sequences, find the length of longest subsequence present in both of them. A subsequence is a sequence that appears in the same relative order, but not necessarily contiguous. For example, “abc”, “abg”, “bdf”, “aeg”, ‘”acefg”, .. etc are subsequences of “abcdefg”. So a string of length n has $$2^n$$ different possible subsequences. LCS for input Sequences “ABCDGH” and “AEDFHR” is “ADH” of length 3.

Let the input sequences be X[0..m-1] and Y[0..n-1] of lengths m and n respectively. And let L(X[0..m-1], Y[0..n-1]) be the length of LCS of the two sequences X and Y. Following is the recursive definition of L(X[0..m-1], Y[0..n-1]).

If last characters of both sequences match (or X[m-1] == Y[n-1]) then
L(X[0..m-1], Y[0..n-1]) = 1 + L(X[0..m-2], Y[0..n-2])

If last characters of both sequences do not match (or X[m-1] != Y[n-1]) then
L(X[0..m-1], Y[0..n-1]) = MAX ( L(X[0..m-2], Y[0..n-1]), L(X[0..m-1], Y[0..n-2])



```python
# A Naive recursive Python implementation of LCS problem
 
def lcs(X, Y, m, n):
 
    if m == 0 or n == 0:
       return 0;
    elif X[m-1] == Y[n-1]:
       return 1 + lcs(X, Y, m-1, n-1);
    else:
       return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n));
```

Time complexity of the above naive recursive approach is $$O(2^n)$$ in worst case and worst case happens when all characters of X and Y mismatch i.e., length of LCS is 0.

If we draw the complete recursion tree, then we can see that there are many subproblems which are solved again and again. So this problem has Overlapping Substructure property and recomputation of same subproblems can be avoided by either using Memoization or Tabulation. Following is a tabulated implementation for the LCS problem.


```python
# Dynamic Programming implementation of LCS problem
 
def lcs(X , Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
 
    # declaring the array for storing the dp values
    L = [[None]*(n+1) for i in xrange(m+1)]
 
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0 :
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j] , L[i][j-1])
 
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]
#end of function lcs
# This code is contributed by Nikhil Kumar Singh(nickzuck_007)
```

Time Complexity of the above implementation is O(mn) which is much better than the worst case time complexity of Naive Recursive implementation.


```python
# Dynamic programming implementation of LCS problem
 
# Returns length of LCS for X[0..m-1], Y[0..n-1] 
def lcs(X, Y, m, n):
    L = [[0 for x in xrange(n+1)] for x in xrange(m+1)]
 
    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1] 
    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
 
    # Following code is used to print LCS
    index = L[m][n]
 
    # Create a character array to store the lcs string
    lcs = [""] * (index+1)
    lcs[index] = "\0"
 
    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:
 
        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i-1] == Y[j-1]:
            lcs[index-1] = X[i-1]
            i-=1
            j-=1
            index-=1
 
        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i-1][j] > L[i][j-1]:
            i-=1
        else:
            j-=1
 
    print "LCS of " + X + " and " + Y + " is " + "".join(lcs) 
# This code is contributed by BHAVYA JAIN
```

#### [Min Cost Path](http://www.geeksforgeeks.org/dynamic-programming-set-6-min-cost-path/)

Given a cost matrix cost()() and a position (m, n) in cost()(), write a function that returns cost of minimum cost path to reach (m, n) from (0, 0). Each cell of the matrix represents a cost to traverse through that cell. Total cost of a path to reach (m, n) is sum of all the costs on that path (including both source and destination). You can only traverse down, right and diagonally lower cells from a given cell, i.e., from a given cell (i, j), cells (i+1, j), (i, j+1) and (i+1, j+1) can be traversed. You may assume that all costs are positive integers.

The path to reach (m, n) must be through one of the 3 cells: (m-1, n-1) or (m-1, n) or (m, n-1). So minimum cost to reach (m, n) can be written as “minimum of the 3 cells plus cost[m][n]”.

minCost(m, n) = min (minCost(m-1, n-1), minCost(m-1, n), minCost(m, n-1)) + cost[m][n]


```python
# Dynamic Programming Python implementation of Min Cost Path
# problem
R = 3
C = 3
 
def minCost(cost, m, n):
 
    # Instead of following line, we can use int tc[m+1][n+1] or
    # dynamically allocate memoery to save space. The following
    # line is used to keep te program simple and make it working
    # on all compilers.
    tc = [[0 for x in range(C)] for x in range(R)]
 
    tc[0][0] = cost[0][0]
 
    # Initialize first column of total cost(tc) array
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
 
    # Initialize first row of tc array
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
 
    # Construct rest of the tc array
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
 
    return tc[m][n]

# This code is contributed by Bhavya Jain
```

#### [0-1 Knapsack Problem](http://www.geeksforgeeks.org/dynamic-programming-set-10-0-1-knapsack-problem/)

Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value in the knapsack. In other words, given two integer arrays val[0..n-1] and wt[0..n-1] which represent values and weights associated with n items respectively. Also given an integer W which represents knapsack capacity, find out the maximum value subset of val[] such that sum of the weights of this subset is smaller than or equal to W. You cannot break an item, either pick the complete item, or don’t pick it (0-1 property).

A simple solution is to consider all subsets of items and calculate the total weight and value of all subsets. Consider the only subsets whose total weight is smaller than W. From all such subsets, pick the maximum value subset.

To consider all subsets of items, there can be two cases for every item: (1) the item is included in the optimal subset, (2) not included in the optimal set.

Therefore, the maximum value that can be obtained from n items is max of following two values.

1) Maximum value obtained by n-1 items and W weight (excluding nth item).

2) Value of nth item plus maximum value obtained by n-1 items and W minus weight of the nth item (including nth item).

If weight of nth item is greater than W, then the nth item cannot be included and case 1 is the only possibility.


```python
# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def knapSack(W, wt, val, n):
    K = [[0 for x in range(W+1)] for x in range(n+1)]
 
    # Build table K[][] in bottom up manner
    for i in range(n+1):
        for w in range(W+1):
            if i==0 or w==0:
                K[i][w] = 0
            elif wt[i-1] <= w:
                K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
 
    return K[n][W]
# This code is contributed by Bhavya Jain
```

#### [Longest Palindromic Subsequence](http://www.geeksforgeeks.org/dynamic-programming-set-12-longest-palindromic-subsequence/)

Given a sequence, find the length of the longest palindromic subsequence in it. For example, if the given sequence is “BBABCBCAB”, then the output should be 7 as “BABCBAB” is the longest palindromic subseuqnce in it. “BBBBB” and “BBCBB” are also palindromic subsequences of the given sequence, but not the longest ones.

Let X[0..n-1] be the input sequence of length n and L(0, n-1) be the length of the longest palindromic subsequence of X[0..n-1].

If last and first characters of X are same, then L(0, n-1) = L(1, n-2) + 2.

Else L(0, n-1) = MAX (L(1, n-1), L(0, n-2)).


```python
# A Dynamic Programming based Python program for LPS problem
# Returns the length of the longest palindromic subsequence in seq
def lps(str):
    n = len(str)
 
    # Create a table to store results of subproblems
    L = [[0 for x in range(n)] for x in range(n)]
 
    # Strings of length 1 are palindrome of length 1
    for i in range(n):
        L[i][i] = 1
 
    # Build the table. Note that the lower diagonal values of table are
    # useless and not filled in the process. The values are filled in a
    # manner similar to Matrix Chain Multiplication DP solution (See
    # http://www.geeksforgeeks.org/dynamic-programming-set-8-matrix-chain-multiplication/
    # cl is length of substring
    for cl in range(2, n+1):
        for i in range(n-cl+1):
            j = i+cl-1
            if str[i] == str[j] and cl == 2:
                L[i][j] = 2
            elif str[i] == str[j]:
                L[i][j] = L[i+1][j-1] + 2
            else:
                L[i][j] = max(L[i][j-1], L[i+1][j]);
 
    return L[0][n-1]
# This code is contributed by Bhavya Jain
```


### Greedy Algorithms Problems

Greedy is an algorithmic paradigm that builds up a solution piece by piece, always choosing the next piece that offers the most obvious and immediate benefit. Greedy algorithms are used for optimization problems. An optimization problem can be solved using Greedy if the problem has the following property: At every step, we can make a choice that looks best at the moment, and we get the optimal solution of the complete problem.

If a Greedy Algorithm can solve a problem, then it generally becomes the best method to solve that problem as the Greedy algorithms are in general more efficient than other techniques like Dynamic Programming. But Greedy algorithms cannot always be applied. For example, Fractional Knapsack problem can be solved using Greedy, but 0-1 Knapsack cannot be solved using Greedy.

General Strategy:

* Make a greedy choice
* Prove that it is a safe move
* Reduce to a subproblem
* Solve the subproblem

##### Covering points by segments

Input: A set of n points x1, ..., xn

Output: The minimum number of segments of unit length needed to cover all the points.

Assume $$x1 \le x2 \le ... \le xn$$

```cpp
PointsCoverSorted(x1, ..., xn)
R <- {}, i <- 1
while i <= n:
    [l, r] <- [xi, xi+1]
    R <- R U {[l, r]}
    i <- i + 1
    while i <= n and xi <= r:
        i <- i+1
    return R
```

##### Fractional Knapsack

```cpp
Knapsack(W, w1, v1, ..., wn, vn)
A <- [0, 0, ..., 0], V <- 0
repeat n times:
    if W = 0:
        return (V, A)
    select i with wi >0 and max vi/wi
    a <- min(wi, W)
    V <- V + a * vi/wi
    wi <- wi - a
    A[i] <- A[i] +a
    W <- W - a
return (V, A)
```

Following are some standard algorithms that are Greedy algorithms.

* Kruskal’s Minimum Spanning Tree (MST): In Kruskal’s algorithm, we create a MST by picking edges one by one. The Greedy Choice is to pick the smallest weight edge that doesn’t cause a cycle in the MST constructed so far.

* Prim’s Minimum Spanning Tree: In Prim’s algorithm also, we create a MST by picking edges one by one. We maintain two sets: set of the vertices already included in MST and the set of the vertices not yet included. The Greedy Choice is to pick the smallest weight edge that connects the two sets.

* Dijkstra’s Shortest Path: The Dijkstra’s algorithm is very similar to Prim’s algorithm. The shortest path tree is built up, edge by edge. We maintain two sets: set of the vertices already included in the tree and the set of the vertices not yet included. The Greedy Choice is to pick the edge that connects the two sets and is on the smallest weight path from source to the set that contains not yet included vertices.

* Huffman Coding: Huffman Coding is a loss-less compression technique. It assigns variable length bit codes to different characters. The Greedy Choice is to assign least bit length code to the most frequent character.

The greedy algorithms are sometimes also used to get an approximation for Hard optimization problems. For example, Traveling Salesman Problem is a NP Hard problem. A Greedy choice for this problem is to pick the nearest unvisited city from the current city at every step. This solutions doesn’t always produce the best optimal solution, but can be used to get an approximate optimal solution.

#### [Activity Selection problem](http://www.geeksforgeeks.org/greedy-algorithms-set-1-activity-selection-problem/)

You are given n activities with their start and finish times. Select the maximum number of activities that can be performed by a single person, assuming that a person can only work on a single activity at a time.

Example:

Consider the following 6 activities. 

     start[]  =  {1, 3, 0, 5, 8, 5};
     
     finish[] =  {2, 4, 6, 7, 9, 9};
     
The maximum set of activities that can be executed 
by a single person is {0, 1, 3, 4}

The greedy choice is to always pick the next activity whose finish time is least among the remaining activities and the start time is more than or equal to the finish time of previously selected activity. We can sort the activities according to their finishing time so that we always consider the next activity as minimum finishing time activity.


```python
"""The following implemenatation assumes that the activities
are already sorted according to their finish time"""
 
"""Prints a maximum set of activities that can be done by a
single person, one at a time"""
# n --> Total number of activities
# s[]--> An array that contains start time of all activities
# f[] --> An array that conatins finish time of all activities
 
def printMaxActivities(s , f ):
    n = len(f)
    print "The following activities are selected"
 
    # The first activity is always selected
    i = 0
    print i,
 
    # Consider rest of the activities
    for j in xrange(n):
 
        # If this activity has start time greater than
        # or equal to the finish time of previously 
        # selected activity, then select it
        if s[j] >= f[i]:
            print j,
            i = j
# This code is contributed by Nikhil Kumar Singh
```

#### [Kruskal's Minimum Spanning Tree Algorithm](http://www.geeksforgeeks.org/greedy-algorithms-set-2-kruskals-minimum-spanning-tree-mst/)

What is Minimum Spanning Tree?

Given a connected and undirected graph, a spanning tree of that graph is a subgraph that is a tree and connects all the vertices together. A single graph can have many different spanning trees. A minimum spanning tree (MST) or minimum weight spanning tree for a weighted, connected and undirected graph is a spanning tree with weight less than or equal to the weight of every other spanning tree. The weight of a spanning tree is the sum of weights given to each edge of the spanning tree.

How many edges does a minimum spanning tree has?

A minimum spanning tree has (V – 1) edges where V is the number of vertices in the given graph.

Below are the steps for finding MST using Kruskal’s algorithm

1. Sort all the edges in non-decreasing order of their weight.

2. Pick the smallest edge. Check if it forms a cycle with the spanning tree 
formed so far. If cycle is not formed, include this edge. Else, discard it.  

3. Repeat step#2 until there are (V-1) edges in the spanning tree.


```cpp
// C++ program for Kruskal's algorithm to find Minimum Spanning Tree
// of a given connected, undirected and weighted graph
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
// a structure to represent a weighted edge in graph
struct Edge
{
    int src, dest, weight;
};
 
// a structure to represent a connected, undirected and weighted graph
struct Graph
{
    // V-> Number of vertices, E-> Number of edges
    int V, E;
 
    // graph is represented as an array of edges. Since the graph is
    // undirected, the edge from src to dest is also edge from dest
    // to src. Both are counted as 1 edge here.
    struct Edge* edge;
};
 
// Creates a graph with V vertices and E edges
struct Graph* createGraph(int V, int E)
{
    struct Graph* graph = (struct Graph*) malloc( sizeof(struct Graph) );
    graph->V = V;
    graph->E = E;
 
    graph->edge = (struct Edge*) malloc( graph->E * sizeof( struct Edge ) );
 
    return graph;
}
 
// A structure to represent a subset for union-find
struct subset
{
    int parent;
    int rank;
};
 
// A utility function to find set of an element i
// (uses path compression technique)
int find(struct subset subsets[], int i)
{
    // find root and make root as parent of i (path compression)
    if (subsets[i].parent != i)
        subsets[i].parent = find(subsets, subsets[i].parent);
 
    return subsets[i].parent;
}
 
// A function that does union of two sets of x and y
// (uses union by rank)
void Union(struct subset subsets[], int x, int y)
{
    int xroot = find(subsets, x);
    int yroot = find(subsets, y);
 
    // Attach smaller rank tree under root of high rank tree
    // (Union by Rank)
    if (subsets[xroot].rank < subsets[yroot].rank)
        subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
        subsets[yroot].parent = xroot;
 
    // If ranks are same, then make one as root and increment
    // its rank by one
    else
    {
        subsets[yroot].parent = xroot;
        subsets[xroot].rank++;
    }
}
 
// Compare two edges according to their weights.
// Used in qsort() for sorting an array of edges
int myComp(const void* a, const void* b)
{
    struct Edge* a1 = (struct Edge*)a;
    struct Edge* b1 = (struct Edge*)b;
    return a1->weight > b1->weight;
}
 
// The main function to construct MST using Kruskal's algorithm
void KruskalMST(struct Graph* graph)
{
    int V = graph->V;
    struct Edge result[V];  // Tnis will store the resultant MST
    int e = 0;  // An index variable, used for result[]
    int i = 0;  // An index variable, used for sorted edges
 
    // Step 1:  Sort all the edges in non-decreasing order of their weight
    // If we are not allowed to change the given graph, we can create a copy of
    // array of edges
    qsort(graph->edge, graph->E, sizeof(graph->edge[0]), myComp);
 
    // Allocate memory for creating V ssubsets
    struct subset *subsets =
        (struct subset*) malloc( V * sizeof(struct subset) );
 
    // Create V subsets with single elements
    for (int v = 0; v < V; ++v)
    {
        subsets[v].parent = v;
        subsets[v].rank = 0;
    }
 
    // Number of edges to be taken is equal to V-1
    while (e < V - 1)
    {
        // Step 2: Pick the smallest edge. And increment the index
        // for next iteration
        struct Edge next_edge = graph->edge[i++];
 
        int x = find(subsets, next_edge.src);
        int y = find(subsets, next_edge.dest);
 
        // If including this edge does't cause cycle, include it
        // in result and increment the index of result for next edge
        if (x != y)
        {
            result[e++] = next_edge;
            Union(subsets, x, y);
        }
        // Else discard the next_edge
    }
 
    // print the contents of result[] to display the built MST
    printf("Following are the edges in the constructed MST\n");
    for (i = 0; i < e; ++i)
        printf("%d -- %d == %d\n", result[i].src, result[i].dest,
                                                   result[i].weight);
    return;
}
 
// Driver program to test above functions
int main()
{
    /* Let us create following weighted graph
             10
        0--------1
        |  \     |
       6|   5\   |15
        |      \ |
        2--------3
            4       */
    int V = 4;  // Number of vertices in graph
    int E = 5;  // Number of edges in graph
    struct Graph* graph = createGraph(V, E);
 
 
    // add edge 0-1
    graph->edge[0].src = 0;
    graph->edge[0].dest = 1;
    graph->edge[0].weight = 10;
 
    // add edge 0-2
    graph->edge[1].src = 0;
    graph->edge[1].dest = 2;
    graph->edge[1].weight = 6;
 
    // add edge 0-3
    graph->edge[2].src = 0;
    graph->edge[2].dest = 3;
    graph->edge[2].weight = 5;
 
    // add edge 1-3
    graph->edge[3].src = 1;
    graph->edge[3].dest = 3;
    graph->edge[3].weight = 15;
 
    // add edge 2-3
    graph->edge[4].src = 2;
    graph->edge[4].dest = 3;
    graph->edge[4].weight = 4;
 
    KruskalMST(graph);
 
    return 0;
}
```

#### [Prim’s Minimum Spanning Tree (MST)](http://www.geeksforgeeks.org/greedy-algorithms-set-5-prims-minimum-spanning-tree-mst-2/)

It starts with an empty spanning tree. The idea is to maintain two sets of vertices. The first set contains the vertices already included in the MST, the other set contains the vertices not yet included. At every step, it considers all the edges that connect the two sets, and picks the minimum weight edge from these edges. After picking the edge, it moves the other endpoint of the edge to the set containing MST.

A group of edges that connects two set of vertices in a graph is called cut in graph theory. So, at every step of Prim’s algorithm, we find a cut (of two sets, one contains the vertices already included in MST and other contains rest of the verices), pick the minimum weight edge from the cut and include this vertex to MST Set (the set that contains already included vertices).

The idea behind Prim’s algorithm is simple, a spanning tree means all vertices must be connected. So the two disjoint subsets (discussed above) of vertices must be connected to make a Spanning Tree. And they must be connected with the minimum weight edge to make it a Minimum Spanning Tree.

Algorithm:

1) Create a set mstSet that keeps track of vertices already included in MST.

2) Assign a key value to all vertices in the input graph. Initialize all key values as INFINITE. Assign key value as 0 for the first vertex so that it is picked first.

3) While mstSet doesn’t include all vertices

    * a) Pick a vertex u which is not there in mstSet and has minimum key value.

    * b) Include u to mstSet.

    * c) Update key value of all adjacent vertices of u. To update the key values, iterate through all adjacent vertices. For every adjacent vertex v, if weight of edge u-v is less than the previous key value of v, update the key value as weight of u-v

The idea of using key values is to pick the minimum weight edge from cut. The key values are used only for vertices which are not yet included in MST, the key value for these vertices indicate the minimum weight edges connecting them to the set of vertices included in MST.

```cpp
// A C / C++ program for Prim's Minimum Spanning Tree (MST) algorithm. 
// The program is for adjacency matrix representation of the graph
 
#include <stdio.h>
#include <limits.h>
 
// Number of vertices in the graph
#define V 5
 
// A utility function to find the vertex with minimum key value, from
// the set of vertices not yet included in MST
int minKey(int key[], bool mstSet[])
{
   // Initialize min value
   int min = INT_MAX, min_index;
 
   for (int v = 0; v < V; v++)
     if (mstSet[v] == false && key[v] < min)
         min = key[v], min_index = v;
 
   return min_index;
}
 
// A utility function to print the constructed MST stored in parent[]
int printMST(int parent[], int n, int graph[V][V])
{
   printf("Edge   Weight\n");
   for (int i = 1; i < V; i++)
      printf("%d - %d    %d \n", parent[i], i, graph[i][parent[i]]);
}
 
// Function to construct and print MST for a graph represented using adjacency
// matrix representation
void primMST(int graph[V][V])
{
     int parent[V]; // Array to store constructed MST
     int key[V];   // Key values used to pick minimum weight edge in cut
     bool mstSet[V];  // To represent set of vertices not yet included in MST
 
     // Initialize all keys as INFINITE
     for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;
 
     // Always include first 1st vertex in MST.
     key[0] = 0;     // Make key 0 so that this vertex is picked as first vertex
     parent[0] = -1; // First node is always root of MST 
 
     // The MST will have V vertices
     for (int count = 0; count < V-1; count++)
     {
        // Pick thd minimum key vertex from the set of vertices
        // not yet included in MST
        int u = minKey(key, mstSet);
 
        // Add the picked vertex to the MST Set
        mstSet[u] = true;
 
        // Update key value and parent index of the adjacent vertices of
        // the picked vertex. Consider only those vertices which are not yet
        // included in MST
        for (int v = 0; v < V; v++)
 
           // graph[u][v] is non zero only for adjacent vertices of m
           // mstSet[v] is false for vertices not yet included in MST
           // Update the key only if graph[u][v] is smaller than key[v]
          if (graph[u][v] && mstSet[v] == false && graph[u][v] <  key[v])
             parent[v]  = u, key[v] = graph[u][v];
     }
 
     // print the constructed MST
     printMST(parent, V, graph);
}
 
 
// driver program to test above function
int main()
{
   /* Let us create the following graph
          2    3
      (0)--(1)--(2)
       |   / \   |
      6| 8/   \5 |7
       | /     \ |
      (3)-------(4)
            9          */
   int graph[V][V] = { {0, 2, 0, 6, 0},{2, 0, 3, 8, 5},{0, 3, 0, 0, 7},{6, 8, 0, 0, 9},{0, 5, 7, 9, 0} };
 
    // Print the solution
    primMST(graph);
 
    return 0;
}
```

#### [Dijkstra’s shortest path algorithm](http://www.geeksforgeeks.org/greedy-algorithms-set-6-dijkstras-shortest-path-algorithm/)

Given a graph and a source vertex in graph, find shortest paths from source to all vertices in the given graph.

Dijkstra’s algorithm is very similar to Prim’s algorithm for minimum spanning tree. Like Prim’s MST, we generate a SPT (shortest path tree) with given source as root. We maintain two sets, one set contains vertices included in shortest path tree, other set includes vertices not yet included in shortest path tree. At every step of the algorithm, we find a vertex which is in the other set (set of not yet included) and has minimum distance from source.

Algorithm:

1) Create a set sptSet (shortest path tree set) that keeps track of vertices included in shortest path tree, i.e., whose minimum distance from source is calculated and finalized. Initially, this set is empty.

2) Assign a distance value to all vertices in the input graph. Initialize all distance values as INFINITE. Assign distance value as 0 for the source vertex so that it is picked first.

3) While sptSet doesn’t include all vertices

    * a) Pick a vertex u which is not there in sptSetand has minimum distance value.

    * b) Include u to sptSet.

    * c) Update distance value of all adjacent vertices of u. To update the distance values, iterate through all adjacent vertices. For every adjacent vertex v, if sum of distance value of u (from source) and weight of edge u-v, is less than the distance value of v, then update the distance value of v.

```cpp
// A C / C++ program for Dijkstra's single source shortest path algorithm.
// The program is for adjacency matrix representation of the graph
 
#include <stdio.h>
#include <limits.h>
 
// Number of vertices in the graph
#define V 9
 
// A utility function to find the vertex with minimum distance value, from
// the set of vertices not yet included in shortest path tree
int minDistance(int dist[], bool sptSet[])
{
   // Initialize min value
   int min = INT_MAX, min_index;
 
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= min)
         min = dist[v], min_index = v;
 
   return min_index;
}
 
// A utility function to print the constructed distance array
int printSolution(int dist[], int n)
{
   printf("Vertex   Distance from Source\n");
   for (int i = 0; i < V; i++)
      printf("%d \t\t %d\n", i, dist[i]);
}
 
// Funtion that implements Dijkstra's single source shortest path algorithm
// for a graph represented using adjacency matrix representation
void dijkstra(int graph[V][V], int src)
{
     int dist[V];     // The output array.  dist[i] will hold the shortest
                      // distance from src to i
 
     bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
                     // path tree or shortest distance from src to i is finalized
 
     // Initialize all distances as INFINITE and stpSet[] as false
     for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
 
     // Distance of source vertex from itself is always 0
     dist[src] = 0;
 
     // Find shortest path for all vertices
     for (int count = 0; count < V-1; count++)
     {
       // Pick the minimum distance vertex from the set of vertices not
       // yet processed. u is always equal to src in first iteration.
       int u = minDistance(dist, sptSet);
 
       // Mark the picked vertex as processed
       sptSet[u] = true;
 
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++)
 
         // Update dist[v] only if is not in sptSet, there is an edge from 
         // u to v, and total weight of path from src to  v through u is 
         // smaller than current value of dist[v]
         if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX 
                                       && dist[u]+graph[u][v] < dist[v])
            dist[v] = dist[u] + graph[u][v];
     }
 
     // print the constructed distance array
     printSolution(dist, V);
}
 
// driver program to test above function
int main()
{
   /* Let us create the example graph discussed above */
   int graph[V][V] = { {0, 4, 0, 0, 0, 0, 0, 8, 0},
                      {4, 0, 8, 0, 0, 0, 0, 11, 0},
                      {0, 8, 0, 7, 0, 4, 0, 0, 2},
                      {0, 0, 7, 0, 9, 14, 0, 0, 0},
                      {0, 0, 0, 9, 0, 10, 0, 0, 0},
                      {0, 0, 4, 0, 10, 0, 2, 0, 0},
                      {0, 0, 0, 14, 0, 2, 0, 1, 6},
                      {8, 11, 0, 0, 0, 0, 1, 0, 7},
                      {0, 0, 2, 0, 0, 0, 6, 7, 0}
                     };
 
    dijkstra(graph, 0);
 
    return 0;
}
```

### [Divide and Conquer](http://www.geeksforgeeks.org/divide-and-conquer-set-1-find-closest-pair-of-points/)

A typical Divide and Conquer algorithm solves a problem using following three steps.

* Divide: Break the given problem into non-overlapping subproblems of same type.
* Conquer: Recursively solve these subproblems
* Combine: Appropriately combine the answers

** Linear Search: Searching in an array ** 

Input: An array A with n elements; a key k.

Output: An index, i, where A[i]=k. If there is no such i, return NOT_FOUND.

```
# Recursive Solution, worst-case $$\Theta(n)$$
LinearSearch(A, low, high, key):
if high < low:
    return NOT_FOUND
if A[low] == key:
    return low
return LinearSearch(A, low+1, high, key)

# Iterative Version
LinearSearchIt(A, low, high, key)
for i from low to high:
    if A[i] == key:
        return i
return NOT_FOUND
```

** Binary Search: Searching in a sorted array **

Input: A sorted array A[low ... high] (low $$\le$$ i < high: A[i] $$\le$$ A[i+1]). A key k.

Output: An index, i, (low $$\le$$ i $$\le$$ high) where A[i] = k. Otherwise, the greatest index i, where A[i] < k. Otherwise (k < A[low]), the result is low - 1.

```
# Binary Search - recursive version: $$\Theta(log(n))$$
BinarySearch(A, low, high, key)
if high < low:
    return low - 1
mid <- floor(low + (high-low)/2)
if key = A[mid]:
    return mid
else if key < A[mid]:
    return BinarySearch(A, low, mid-1, key)
else:
    return BinarySearch(A, mid + 1, high, key)

# Iterative Version
BinarySearchIt(A, low, high, key)
while low <= high:
    mid <- floor(low + (high-low)/2)
    if key = A[mid]:
        return mid
    else if key < A[mid]:
        high = mid - 1
    else:
        low = mid + 1
return low - 1
```

** Multiplying polynomials **

Input: Two n-1 degree polynomials $$a_{n-1}x^{n-1} + a_{n-2}x^{n-2} + \ldots + a_1 x+a_0$$, $$b_{n-1}x^{n-1} + b_{n-2}x^{n-2}+\ldots+b_1x+b_0$$

Output: The product polynomial $$c_{2n-2}x^{2n-2} +c_{2n-3}x^{2n-3} +\ldots+c_1x+c_0$$, where $$c_{2n-2} = a_{n-1}b_{n-1}, c_{2n-3}=a_{n-1}b_{n-2}+a_{n-2}b_{n-1}, \ldots, c_2 = a_2b_0+a_1b_1+a_0b_2, c_1=a_1b_0+a_0b_1, c_0 = a_0b_0$$.

```
# Naive polynomial multiplying $$O(n^2)$$
MultPoly(A, B, n)
pair = Array[n][n]
for i = 0 to n-1:
    for j = 0 to n-1:
        pair[i][j] = A[i]*B[j]
product = Array[2n-1]
for i = 0 to 2n-1:
    product[i] = 0
for i = 0 to n-1:
    for j=i to n-1:
        product[i+j] = product[i+j] + pair[i][j]
return product
```

Naive Divide and Conquer:

Let $$ A(x) = D_1(x) x^{n/2} + D_0(x)$$ where $$D_1(x) = a_{n-1}x^{n/2-1}+a_{n-2}x^{n/2-2} + \ldots + a_{n/2}, D_0(x) = a_{n/2-1}x^{n/2-1}+a_{n/2-2}x^{n/2-1}+\ldots+a_0$$, let $$B(x) = E_1(x)x^{n/2}+E_0(x)$$, $$E_1(x)$$ and $$E_0(x)$$ is similar as $$D_1(x)$$ and $$D_0(x)$$, then $$AB=(D_1x^{n/2}+D_0)(E_1x^{n/2}+E_0)

```
# Naive Divide and Conquer for polynomial multiplying $$\Theta(n^2)$$
Function Mult2(A, B, n, al, bl)
R = array[0...2n-2]
if n = 1:
    R[0] = A[al] * B[bl]
    return R
R[0...n-2] = Mult2(A, B, n/2, al, bl)
R[n...2n-2] = Mult2(A, B, n/2, al+n/2, bl+n/2)
D0E1 = Mult2(A, B, n/2, al, bl+n/2)
D1E0 = Mult2(A, B, n/2, al+n/2, b1)
R[n/2...n+n/2-2] += D1E0 + D0E1
```

Karatsuba approach:

$$ A(x)= a_1 x + a_0$$, $$B(x) = b_1x+b_0$$, $$C(x) = a_1 b_1 x^2 + (a_1 b_0 + a_0 b_1) x + a_0 b_0$$, needs 4 multiplicaitons.

Rewrite as $$ C(x) = a_1 b_1 x^2 + ((a_1 + a_0)(b_1 + b_0) -a_1 b_1 -a_0 b_0)x+ a_0 b_0$$ needs 3 multiplications.



Following are some standard algorithms that are Divide and Conquer algorithms.

1) Binary Search is a searching algorithm. In each step, the algorithm compares the input element x with the value of the middle element in array. If the values match, return the index of middle. Otherwise, if x is less than the middle element, then the algorithm recurs for left side of middle element, else recurs for right side of middle element.

2) Quicksort is a sorting algorithm. The algorithm picks a pivot element, rearranges the array elements in such a way that all elements smaller than the picked pivot element move to left side of pivot, and all greater elements move to right side. Finally, the algorithm recursively sorts the subarrays on left and right of pivot element.

3) Merge Sort is also a sorting algorithm. The algorithm divides the array in two halves, recursively sorts them and finally merges the two sorted halves.

4) Closest Pair of Points The problem is to find the closest pair of points in a set of points in x-y plane. The problem can be solved in $$O(n^2)$$ time by calculating distances of every pair of points and comparing the distances to find the minimum. The Divide and Conquer algorithm solves the problem in O(nLogn) time.

5) Strassen’s Algorithm is an efficient algorithm to multiply two matrices. A simple method to multiply two matrices need 3 nested loops and is $$O(n^3)$$. Strassen’s algorithm multiplies two matrices in $O(n^2.8974)$ time.

6) Cooley–Tukey Fast Fourier Transform (FFT) algorithm is the most common algorithm for FFT. It is a divide and conquer algorithm which works in O(nlogn) time.

7) Karatsuba algorithm for fast multiplication  it does multiplication of two n-digit numbers in at most $$3 n^{\log_23}\approx 3 n^{1.585}$$ single-digit multiplications in general (and exactly $$n^{\log_23}$$ when n is a power of 2). It is therefore faster than the classical algorithm, which requires n2 single-digit products. If n = 210 = 1024, in particular, the exact counts are 310 = 59,049 and (210)2 = 1,048,576, respectively.

We will publishing above algorithms in separate posts.

Divide and Conquer (D & C) vs Dynamic Programming (DP)
Both paradigms (D & C and DP) divide the given problem into subproblems and solve subproblems. How to choose one of them for a given problem? Divide and Conquer should be used when same subproblems are not evaluated many times. Otherwise Dynamic Programming or Memoization should be used. For example, Binary Search is a Divide and Conquer algorithm, we never evaluate the same subproblems again. On the other hand, for calculating nth Fibonacci number, Dynamic Programming should be preferred.

```cpp
# Write a program to calculate pow(x,n)
/* Function to calculate x raised to the power y in O(logn)*/
int power(int x, unsigned int y)
{
    int temp;
    if( y == 0)
        return 1;
    temp = power(x, y/2);
    if (y%2 == 0)
        return temp*temp;
    else
        return x*temp*temp;
}
```

#### [Median of two sorted arrays](http://www.geeksforgeeks.org/median-of-two-sorted-arrays/)

Question: There are 2 sorted arrays A and B of size n each. Write an algorithm to find the median of the array obtained after merging the above 2 arrays(i.e. array of length 2n). The complexity should be O(log(n))

The median of a finite list of numbers can be found by arranging all the numbers from lowest value to highest value and picking the middle one.

Let us see different methods to get the median of two sorted arrays of size n each. Since size of the set for which we are looking for median is even (2n), we are taking average of middle two numbers in all below solutions.

**By comparing the medians of two arrays** 
This method works by first getting medians of the two sorted arrays and then comparing them.

Let ar1 and ar2 be the input arrays.

Algorithm:

* Calculate the medians m1 and m2 of the input arrays ar1[] and ar2[] respectively.

* If m1 and m2 both are equal then we are done.
     return m1 (or m2)
     
* If m1 is greater than m2, then median is present in one of the below two subarrays.
    * From first element of ar1 to m1 (ar1[0...n/2])
    * From m2 to last element of ar2  (ar2[n/2...n-1])
* If m2 is greater than m1, then median is present in one
   of the below two subarrays.
    * From m1 to last element of ar1  (ar1[n/2...n-1])
    * From first element of ar2 to m2 (ar2[0...n/2])
* Repeat the above process until size of both the subarrays becomes 2.
* If size of the two arrays is 2 then use below formula to get the median.
    Median = (max(ar1[0], ar2[0]) + min(ar1[1], ar2[1]))/2

```cpp
#include<stdio.h>
 
int max(int, int); /* to get maximum of two integers */
int min(int, int); /* to get minimum of two integeres */
int median(int [], int); /* to get median of a sorted array */
 
/* This function returns median of ar1[] and ar2[].
   Assumptions in this function:
   Both ar1[] and ar2[] are sorted arrays
   Both have n elements */
int getMedian(int ar1[], int ar2[], int n)
{
    int m1; /* For median of ar1 */
    int m2; /* For median of ar2 */
 
    /* return -1  for invalid input */
    if (n <= 0)
        return -1;
 
    if (n == 1)
        return (ar1[0] + ar2[0])/2;
 
    if (n == 2)
        return (max(ar1[0], ar2[0]) + min(ar1[1], ar2[1])) / 2;
 
    m1 = median(ar1, n); /* get the median of the first array */
    m2 = median(ar2, n); /* get the median of the second array */
 
    /* If medians are equal then return either m1 or m2 */
    if (m1 == m2)
        return m1;
 
     /* if m1 < m2 then median must exist in ar1[m1....] and ar2[....m2] */
    if (m1 < m2)
    {
        if (n % 2 == 0)
            return getMedian(ar1 + n/2 - 1, ar2, n - n/2 +1);
        else
            return getMedian(ar1 + n/2, ar2, n - n/2);
    }
 
    /* if m1 > m2 then median must exist in ar1[....m1] and ar2[m2...] */
    else
    {
        if (n % 2 == 0)
            return getMedian(ar2 + n/2 - 1, ar1, n - n/2 + 1);
        else
            return getMedian(ar2 + n/2, ar1, n - n/2);
    }
}
 
/* Function to get median of a sorted array */
int median(int arr[], int n)
{
    if (n%2 == 0)
        return (arr[n/2] + arr[n/2-1])/2;
    else
        return arr[n/2];
}
 
/* Driver program to test above function */
int main()
{
    int ar1[] = {1, 2, 3, 6};
    int ar2[] = {4, 6, 8, 10};
    int n1 = sizeof(ar1)/sizeof(ar1[0]);
    int n2 = sizeof(ar2)/sizeof(ar2[0]);
    if (n1 == n2)
      printf("Median is %d", getMedian(ar1, ar2, n1));
    else
     printf("Doesn't work for arrays of unequal size");
 
    getchar();
    return 0;
}
 
/* Utility functions */
int max(int x, int y)
{
    return x > y? x : y;
}
 
int min(int x, int y)
{
    return x > y? y : x;
}
```

**By doing binary search for the median**

The basic idea is that if you are given two arrays ar1[] and ar2[] and know the length of each, you can check whether an element ar1[i] is the median in constant time. Suppose that the median is ar1[i]. Since the array is sorted, it is greater than exactly i values in array ar1[]. Then if it is the median, it is also greater than exactly j = n – i – 1 elements in ar2[].

It requires constant time to check if ar2[j] <= ar1[i] <= ar2[j + 1]. If ar1[i] is not the median, then depending on whether ar1[i] is greater or less than ar2[j] and ar2[j + 1], you know that ar1[i] is either greater than or less than the median. Thus you can binary search for median in O(lg n) worst-case time. For two arrays ar1 and ar2, first do binary search in ar1[]. If you reach at the end (left or right) of the first array and don't find median, start searching in the second array ar2[].

* Get the middle element of ar1[] using array indexes left and right.  
   Let index of the middle element be i.
* Calculate the corresponding index j of ar2[]
     j = n – i – 1 
* If ar1[i] >= ar2[j] and ar1[i] <= ar2[j+1] then ar1[i] and ar2[j]
   are the middle elements.
     return average of ar2[j] and ar1[i]
* If ar1[i] is greater than both ar2[j] and ar2[j+1] then 
     do binary search in left half  (i.e., arr[left ... i-1])
* If ar1[i] is smaller than both ar2[j] and ar2[j+1] then
     do binary search in right half (i.e., arr[i+1....right])
* If you reach at any corner of ar1[] then do binary search in ar2[]

```cpp
#include<stdio.h>
 
int getMedianRec(int ar1[], int ar2[], int left, int right, int n);
 
/* This function returns median of ar1[] and ar2[].
   Assumptions in this function:
   Both ar1[] and ar2[] are sorted arrays
   Both have n elements */
int getMedian(int ar1[], int ar2[], int n)
{
    return getMedianRec(ar1, ar2, 0, n-1, n);
}
 
/* A recursive function to get the median of ar1[] and ar2[]
   using binary search */
int getMedianRec(int ar1[], int ar2[], int left, int right, int n)
{
    int i, j;
 
    /* We have reached at the end (left or right) of ar1[] */
    if (left > right)
        return getMedianRec(ar2, ar1, 0, n-1, n);
 
    i = (left + right)/2;
    j = n - i - 1;  /* Index of ar2[] */
 
    /* Recursion terminates here.*/
    if (ar1[i] > ar2[j] && (j == n-1 || ar1[i] <= ar2[j+1]))
    {
        /* ar1[i] is decided as median 2, now select the median 1
           (element just before ar1[i] in merged array) to get the
           average of both*/
        if (i == 0 || ar2[j] > ar1[i-1])
            return (ar1[i] + ar2[j])/2;
        else
            return (ar1[i] + ar1[i-1])/2;
    }
 
    /*Search in left half of ar1[]*/
    else if (ar1[i] > ar2[j] && j != n-1 && ar1[i] > ar2[j+1])
        return getMedianRec(ar1, ar2, left, i-1, n);
 
    /*Search in right half of ar1[]*/
    else /* ar1[i] is smaller than both ar2[j] and ar2[j+1]*/
        return getMedianRec(ar1, ar2, i+1, right, n);
}
 
/* Driver program to test above function */
int main()
{
    int ar1[] = {1, 12, 15, 26, 38};
    int ar2[] = {2, 13, 17, 30, 45};
    int n1 = sizeof(ar1)/sizeof(ar1[0]);
    int n2 = sizeof(ar2)/sizeof(ar2[0]);
    if (n1 == n2)
        printf("Median is %d", getMedian(ar1, ar2, n1));
    else
        printf("Doesn't work for arrays of unequal size");
 
    getchar();
    return 0;
}
```

#### [Closest Pair of Points](http://www.geeksforgeeks.org/closest-pair-of-points/)

We are given an array of n points in the plane, and the problem is to find out the closest pair of points in the array. 

The Brute force solution is $$O(n^2)$$, compute the distance between each pair and return the smallest. We can calculate the smallest distance in O(nlogn) time using Divide and Conquer strategy.

In this post, a $$O(n(Logn)^2)$$ approach is discussed. We will be discussing a O(nLogn) approach in a separate post.

As a pre-processing step, input array is sorted according to x coordinates.

1) Find the middle point in the sorted array, we can take P[n/2] as middle point.

2) Divide the given array in two halves. The first subarray contains points from P[0] to P[n/2]. The second subarray contains points from P[n/2+1] to P[n-1].

3) Recursively find the smallest distances in both subarrays. Let the distances be dl and dr. Find the minimum of dl and dr. Let the minimum be d.

4) From above 3 steps, we have an upper bound d of minimum distance. Now we need to consider the pairs such that one point in pair is from left half and other is from right half. Consider the vertical line passing through passing through P[n/2] and find all points whose x coordinate is closer than d to the middle vertical line. Build an array strip[] of all such points.

5) Sort the array strip[] according to y coordinates. This step is O(nLogn). It can be optimized to O(n) by recursively sorting and merging.

6) Find the smallest distance in strip[]. This is tricky. From first look, it seems to be a $$O(n^2)$$ step, but it is actually O(n). It can be proved geometrically that for every point in strip, we only need to check at most 7 points after it (note that strip is sorted according to Y coordinate). See this for more analysis.

7) Finally return the minimum of d and distance calculated in above step (step 6)

```cpp
// A divide and conquer program in C/C++ to find the smallest distance from a
// given set of points.
 
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
 
// A structure to represent a Point in 2D plane
struct Point
{
    int x, y;
};
 
/* Following two functions are needed for library function qsort().
   Refer: http://www.cplusplus.com/reference/clibrary/cstdlib/qsort/ */
 
// Needed to sort array of points according to X coordinate
int compareX(const void* a, const void* b)
{
    Point *p1 = (Point *)a,  *p2 = (Point *)b;
    return (p1->x - p2->x);
}
// Needed to sort array of points according to Y coordinate
int compareY(const void* a, const void* b)
{
    Point *p1 = (Point *)a,   *p2 = (Point *)b;
    return (p1->y - p2->y);
}
 
// A utility function to find the distance between two points
float dist(Point p1, Point p2)
{
    return sqrt( (p1.x - p2.x)*(p1.x - p2.x) +
                 (p1.y - p2.y)*(p1.y - p2.y)
               );
}
 
// A Brute Force method to return the smallest distance between two points
// in P[] of size n
float bruteForce(Point P[], int n)
{
    float min = FLT_MAX;
    for (int i = 0; i < n; ++i)
        for (int j = i+1; j < n; ++j)
            if (dist(P[i], P[j]) < min)
                min = dist(P[i], P[j]);
    return min;
}
 
// A utility function to find minimum of two float values
float min(float x, float y)
{
    return (x < y)? x : y;
}
 
 
// A utility function to find the distance beween the closest points of
// strip of given size. All points in strip[] are sorted accordint to
// y coordinate. They all have an upper bound on minimum distance as d.
// Note that this method seems to be a O(n^2) method, but it's a O(n)
// method as the inner loop runs at most 6 times
float stripClosest(Point strip[], int size, float d)
{
    float min = d;  // Initialize the minimum distance as d
 
    qsort(strip, size, sizeof(Point), compareY); 
 
    // Pick all points one by one and try the next points till the difference
    // between y coordinates is smaller than d.
    // This is a proven fact that this loop runs at most 6 times
    for (int i = 0; i < size; ++i)
        for (int j = i+1; j < size && (strip[j].y - strip[i].y) < min; ++j)
            if (dist(strip[i],strip[j]) < min)
                min = dist(strip[i], strip[j]);
 
    return min;
}
 
// A recursive function to find the smallest distance. The array P contains
// all points sorted according to x coordinate
float closestUtil(Point P[], int n)
{
    // If there are 2 or 3 points, then use brute force
    if (n <= 3)
        return bruteForce(P, n);
 
    // Find the middle point
    int mid = n/2;
    Point midPoint = P[mid];
 
    // Consider the vertical line passing through the middle point
    // calculate the smallest distance dl on left of middle point and
    // dr on right side
    float dl = closestUtil(P, mid);
    float dr = closestUtil(P + mid, n-mid);
 
    // Find the smaller of two distances
    float d = min(dl, dr);
 
    // Build an array strip[] that contains points close (closer than d)
    // to the line passing through the middle point
    Point strip[n];
    int j = 0;
    for (int i = 0; i < n; i++)
        if (abs(P[i].x - midPoint.x) < d)
            strip[j] = P[i], j++;
 
    // Find the closest points in strip.  Return the minimum of d and closest
    // distance is strip[]
    return min(d, stripClosest(strip, j, d) );
}
 
// The main functin that finds the smallest distance
// This method mainly uses closestUtil()
float closest(Point P[], int n)
{
    qsort(P, n, sizeof(Point), compareX);
 
    // Use recursive function closestUtil() to find the smallest distance
    return closestUtil(P, n);
}
 
// Driver program to test above functions
int main()
{
    Point P[] = { {2, 3}, {12, 30}, {40, 50}, {5, 1}, {12, 10}, {3, 4}};
    int n = sizeof(P) / sizeof(P[0]);
    printf("The smallest distance is %f ", closest(P, n));
    return 0;
}
```

#### [Strassen’s Matrix Multiplication](http://www.geeksforgeeks.org/strassens-matrix-multiplication/)

Given two square matrices A and B of size n x n each, find their multiplication matrix.

Following is a simple way to multiply two matrices.

```cpp
void multiply(int A[][N], int B[][N], int C[][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < N; k++)
            {
                C[i][j] += A[i][k]*B[k][j];
            }
        }
    }
}
```
Time Complexity of above method is $$O(N^3)$$.

Following is simple Divide and Conquer method to multiply two square matrices.

1) Divide matrices A and B in 4 sub-matrices of size N/2 x N/2 as shown in the below diagram.

2) Calculate following values recursively. ae + bg, af + bh, ce + dg and cf + dh.
![Matrix divide and conquer](http://d1gjlxt8vb0knt.cloudfront.net//wp-content/uploads/strassen_new.png)

In the above method, we do 8 multiplications for matrices of size N/2 x N/2 and 4 additions. Addition of two matrices takes $$O(N^2)$$ time. So the time complexity can be written as

T(N) = 8T(N/2) + $$O(N^2)$$

From Master's Theorem, time complexity of above method is $$O(N^3)$$
which is unfortunately same as the above naive method.

Strassen’s method is similar to above simple divide and conquer method in the sense that this method also divide matrices to sub-matrices of size N/2 x N/2 as shown in the above diagram, but in Strassen’s method, the four sub-matrices of result are calculated using following formulae.

![Strassens](http://d1gjlxt8vb0knt.cloudfront.net//wp-content/uploads/stressen_formula_new_new.png)

Time Complexity of Strassen’s Method
Addition and Subtraction of two matrices takes $$O(N^2)$$ time. So time complexity can be written as

T(N) = 7T(N/2) +  $$O(N^2)$$

From Master's Theorem, time complexity of above method is 
$$O(N^{Log7})$$ which is approximately $$O(N^{2.8074})$$.

Generally Strassen’s Method is not preferred for practical applications for following reasons.

1) The constants used in Strassen’s method are high and for a typical application Naive method works better.

2) For Sparse matrices, there are better methods especially designed for them.

3) The submatrices in recursion take extra space.

4) Because of the limited precision of computer arithmetic on noninteger values, larger errors accumulate in Strassen’s algorithm than in Naive Method.

#### [Binary Search](http://geeksquiz.com/binary-search/)

Given a sorted array arr[] of n elements, write a function to search a given element x in arr[].

The idea of binary search is to use the information that the array is sorted and reduce the time complexity to O(Logn). We basically ignore half of the elements just after one comparison.

1) Compare x with the middle element.

2) If x matches with middle element, we return the mid index.

3) Else If x is greater than the mid element, then x can only lie in right half subarray after the mid element. So we recur for right half.

4) Else (x is smaller) recur for the left half.

```python
# Python Program for recursive binary search.
 
# Returns index of x in arr if present, else -1
def binarySearch (arr, l, r, x):
 
    # Check base case
    if r >= l:
 
        mid = l + (r - l)/2
 
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
         
        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binarySearch(arr, l, mid-1, x)
 
        # Else the element can only be present in right subarray
        else:
            return binarySearch(arr, mid+1, r, x)
 
    else:
        # Element is not present in the array
        return -1


# Iterative Binary Search Function
# It returns location of x in given array arr if present,
# else returns -1
def binarySearch(arr, l, r, x):
 
    while l<=r:
 
        mid = l + (r - l)/2;
         
        # Check if x is present at mid
        if arr[mid] == x:
            return mid
 
        # If x is greater, ignore left half
        elif arr[mid] < x:
            l = mid + 1
 
        # If x is smaller, ignore right half
        else:
            r = mid - 1
     
    # If we reach here, then the element was not present
    return -1
```

#### Selection Sort

Algorithms $$O(n^2)$$:

* Find a minimum by scanning the array
* Swap it with the first element
* Repeat with the remaining part of the array

Pseudo code:

```python
SelectionSort(A[1...n])
for i from 1 to n:
   minIndex <- i
   for j from i+1 to n:
      if A[j] < A[minIndex]:
         minIndex <- j
   swap(A[i], A[minIndex])
```

#### [Merge Sort](http://geeksquiz.com/merge-sort/)

MergeSort is a Divide and Conquer algorithm. It divides input array in two halves, calls itself for the two halves and then merges the two sorted halves. The merg() function is used for merging two halves. The merge(arr, l, m, r) is key process that assumes that arr[l..m] and arr[m+1..r] are sorted and merges the two sorted sub-arrays into one.

Pseudo code ($$O(nlogn)$$:

```Python
MergeSort(A[1...n])
if n=1:
   return A
m <- [n/2]
B <- MergeSort(A[1...m])
C <- MergeSort(A[m+1...n])
A'<- Merge(B,C)
return A'
```

Pseudo code (for merging two sorted arrays):

```python
Merge(B[1...p], C[1...q])
D <- empty array of size p+q
while B and C are both non-empty:
   b <- the first element of B
   c <- the first element of C
   if b <= c:
      move b from B to the end of D
   else:
      move c from C to the end of D
move the rest of B and C to the end of D
return D
```

```cpp
/* C program for merge sort */
#include<stdlib.h>
#include<stdio.h>
 
/* Function to merge the two haves arr[l..m] and arr[m+1..r] of array arr[] */
void merge(int arr[], int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;
 
    /* create temp arrays */
    int L[n1], R[n2];
 
    /* Copy data to temp arrays L[] and R[] */
    for(i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for(j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];
 
    /* Merge the temp arrays back into arr[l..r]*/
    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
 
    /* Copy the remaining elements of L[], if there are any */
    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }
 
    /* Copy the remaining elements of R[], if there are any */
    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }
}
 
/* l is for left index and r is right index of the sub-array
  of arr to be sorted */
void mergeSort(int arr[], int l, int r)
{
    if (l < r)
    {
        int m = l+(r-l)/2; //Same as (l+r)/2, but avoids overflow for large l and h
        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}
 
 
/* UITLITY FUNCTIONS */
/* Function to print an array */
void printArray(int A[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", A[i]);
    printf("\n");
}
 
/* Driver program to test above functions */
int main()
{
    int arr[] = {12, 11, 13, 5, 6, 7};
    int arr_size = sizeof(arr)/sizeof(arr[0]);
 
    printf("Given array is \n");
    printArray(arr, arr_size);
 
    mergeSort(arr, 0, arr_size - 1);
 
    printf("\nSorted array is \n");
    printArray(arr, arr_size);
    return 0;
}
```

#### Counting Sort

Ideas:

* Assume that all elements of A[1...n] are integers from 1 to M.
* By a single scan of the array A, count the number of occurrences of each 1 <= k <= M in the array A and sotre it in Count [k].
* Using this information, fill in the sorted array A'.

Pseudo code $$O(n+M)$$:

```Python
CountSort(A[1...n])
Count[1...M] <- [0,...,0]
for i from i to n:
   Count[A[i]] <- Count[A[i]] + 1
Pos[1...M] <- [0,...,0]
Pos[1] <- 1
for j from 2 to M:
   Pos[j] <- Pos[j-1] + Count[j-1]
for i from 1 to n:
   A'[Pos[A[i]]] <- A[i]
   Pos[A[i]] <- Pos[A[i]] + 1
```

Any comparison based sorting algorithm performs $$\Omega(nlogn)$$ comparisons in the worst case to sort n objects.

#### [QuickSort](http://geeksquiz.com/quick-sort/)

Like Merge Sort, QuickSort is a Divide and Conquer algorithm. It picks an element as pivot and partitions the given array around the picked pivot. There are many different versions of quickSort that pick pivot in different ways.

1) Always pick first element as pivot.

2) Always pick last element as pivot (implemented below)

3) Pick a random element as pivot.

4) Pick median as pivot.

The key process in quickSort is partition(). Target of partitions is, given an array and an element x of array as pivot, put x at its correct position in sorted array and put all smaller elements (smaller than x) before x, and put all greater elements (greater than x) after x. All this should be done in linear time.

Pseudo code for partition:

```Python
Partition(A,l,r)
x <- A[l]
j <- l
for i from l+1 to r:
   if A[i] <= x:
      j <- j+1
      swap A[j] and A[i]
swap A[l] and A[j]
return j
```
Random Pivot:

```Python
RandomizedQuickSort(A, l, r)
if l >= r:
   return
k <- random number between l and r
swap A[l] and A[k]
m <- Parition(A,l,r)
RandomizedQuickSort(A, l, m-1)
RandomizedQuickSort(A, m+1, r)
```
Quick Sort pseudo code:

```python
QuickSort(A, l, r)
while l < r:
   m <- Partition(A, l, r)
   if (m-l) < (r-m):
      QuickSort(A,l,m-1)
      l <- m+1
   else:
      QuickSort(A, m+1, r)
      r <- m-1
```

If all the elements of the given array are equal to each other, then it's $$\Theta(n^2)$$.

To handle equal elements, we can replace the 

```python
m <- Partition (A, l, r)
```

with the line

```python
(m1, m2) <- Partition3(A, l, r)
```

such that

* for all l <= k <= m1-1, A[k] < x
* for all m1 <= k <= m2, A[k] = x
* for all m1+1 <= k <= r, A[k] > x

Pseudo code:
```python
RandomizedQuickSort(A,l,r)
if l >= r:
   return
k <- random number between l and r
swap A[l] and A[k]
(m1, m2) <- Partition3(A,l,r)
RandomizedQuickSort(A,l,m1-1)
RandomizedQuickSort(A,m2+1,r)
```

Partition Algorithm:

There can be many ways to do partition, following code adopts the method given in CLRS book. The logic is simple, we start from the leftmost element and keep track of index of smaller (or equal to) elements as i. While traversing, if we find a smaller element, we swap current element with arr[i]. Otherwise we ignore current element.

```cpp
/* A typical recursive implementation of quick sort */
#include<stdio.h>
 
// A utility function to swap two elements
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}
 
/* This function takes last element as pivot, places the pivot element at its
   correct position in sorted array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right of pivot */
int partition (int arr[], int l, int h)
{
    int x = arr[h];    // pivot
    int i = (l - 1);  // Index of smaller element
 
    for (int j = l; j <= h- 1; j++)
    {
        // If current element is smaller than or equal to pivot 
        if (arr[j] <= x)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);  // Swap current element with index
        }
    }
    swap(&arr[i + 1], &arr[h]);  
    return (i + 1);
}
 
/* arr[] --> Array to be sorted, l  --> Starting index, h  --> Ending index */
void quickSort(int arr[], int l, int h)
{
    if (l < h)
    {
        int p = partition(arr, l, h); /* Partitioning index */
        quickSort(arr, l, p - 1);
        quickSort(arr, p + 1, h);
    }
}
 
/* Function to print an array */
void printArray(int arr[], int size)
{
    int i;
    for (i=0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n");
}
 
// Driver program to test above functions
int main()
{
    int arr[] = {10, 7, 8, 9, 1, 5};
    int n = sizeof(arr)/sizeof(arr[0]);
    quickSort(arr, 0, n-1);
    printf("Sorted array: \n");
    printArray(arr, n);
    return 0;
}
```

#### [Closest pair from two sorted arrays](http://www.geeksforgeeks.org/given-two-sorted-arrays-number-x-find-pair-whose-sum-closest-x/)

Given two sorted arrays and a number x, find the pair whose sum is closest to x and the pair has an element from each array.

We are given two arrays ar1[0…m-1] and ar2[0..n-1] and a number x, we need to find the pair ar1[i] + ar2[j] such that absolute value of (ar1[i] + ar2[j] – x) is minimum.

* Initialize a variable diff as infinite (Diff is used to store the difference between pair and x).  We need to find the minimum diff.
   
* Initialize two index variables l and r in the given sorted array.
    * Initialize first to the leftmost index in ar1:  l = 0
    * Initialize second  the rightmost index in ar2:  r = n-1
       
* Loop while l < m and r >= 0
    * If  abs(ar1[l] + ar2[r] - sum) < diff  then update diff and result 
    * Else if(ar1[l] + ar2[r] <  sum )  then l++
    * Else r - -  
       
* Print the result. 

```cpp
// C++ program to find the pair from two sorted arays such
// that the sum of pair is closest to a given number x
#include <iostream>
#include <climits>
#include <cstdlib>
using namespace std;
 
// ar1[0..m-1] and ar2[0..n-1] are two given sorted arrays
// and x is given number. This function prints the pair  from
// both arrays such that the sum of the pair is closest to x.
void printClosest(int ar1[], int ar2[], int m, int n, int x)
{
    // Initialize the diff between pair sum and x.
    int diff = INT_MAX;
 
    // res_l and res_r are result indexes from ar1[] and ar2[]
    // respectively
    int res_l, res_r;
 
    // Start from left side of ar1[] and right side of ar2[]
    int l = 0, r = n-1;
    while (l<m && r>=0)
    {
       // If this pair is closer to x than the previously
       // found closest, then update res_l, res_r and diff
       if (abs(ar1[l] + ar2[r] - x) < diff)
       {
           res_l = l;
           res_r = r;
           diff = abs(ar1[l] + ar2[r] - x);
       }
 
       // If sum of this pair is more than x, move to smaller
       // side
       if (ar1[l] + ar2[r] > x)
           r--;
       else  // move to the greater side
           l++;
    }
 
    // Print the result
    cout << "The closest pair is [" << ar1[res_l] << ", "
         << ar2[res_r] << "] \n";
}
 
// Driver program to test above functions
int main()
{
    int ar1[] = {1, 4, 5, 7};
    int ar2[] = {10, 20, 30, 40};
    int m = sizeof(ar1)/sizeof(ar1[0]);
    int n = sizeof(ar2)/sizeof(ar2[0]);
    int x = 38;
    printClosest(ar1, ar2, m, n, x);
    return 0;
}
```

[Quick Sort in Python](http://interactivepython.org/runestone/static/pythonds/SortSearch/TheQuickSort.html) 



# Notes for [Technical Interview at Udacity](https://www.udacity.com/course/technical-interview--ud513)


### List-based collections


Python [Time Complexity](https://wiki.python.org/moin/TimeComplexity)

** Linked List**

```python
class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def get_position(self, position):
        counter = 1
        current = self.head
        if position < 1:
            return None
        while current and counter <= position:
            if counter == position:
                return current
            current = current.next
            counter += 1
        return None

    def insert(self, new_element, position):
        counter = 1
        current = self.head
        if position > 1:
            while current and counter < position:
                if counter == position - 1:
                    new_element.next = current.next
                    current.next = new_element
                current = current.next
                counter += 1
        elif position == 1:
            new_element.next = self.head
            self.head = new_element

    def delete(self, value):
        current = self.head
        previous = None
        while current.value != value and current.next:
            previous = current
            current = current.next
        if current.value == value:
            if previous:
                previous.next = current.next
            else:
                self.head = current.next

```

**Stack**

```python
class Element(object):
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList(object):
    def __init__(self, head=None):
        self.head = head

    def append(self, new_element):
        current = self.head
        if self.head:
            while current.next:
                current = current.next
            current.next = new_element
        else:
            self.head = new_element

    def insert_first(self, new_element):
        new_element.next = self.head
        self.head = new_element

    def delete_first(self):
        if self.head:
            deleted_element = self.head
            temp = deleted_element.next
            self.head = temp
            return deleted_element
        else:
            return None

class Stack(object):
    def __init__(self,top=None):
        self.ll = LinkedList(top)

    def push(self, new_element):
        self.ll.insert_first(new_element)

    def pop(self):
        return self.ll.delete_first()

```


**deque**

Adeque is a double-ended queue. You can enqueue on either end. In python, you can import deque by `from collections import deque`.

```python
class Queue(object):
    def __init__(self, head=None):
        self.storage = [head]

    def enqueue(self, new_element):
        self.storage.append(new_element)

    def peek(self):
        return self.storage[0]

    def dequeue(self):
        return self.storage.pop(0)
```

### Searching and Sorting

Binary Search: $O(log(n))$

```python
def binary_search(input_array, value):
    """binary search."""
    first = 0 
    last = len(input_array) - 1
    
    while first <= last:
        mid = (first + last) / 2 
        if input_array[mid] == value:
            return mid
        else:
            if value < input_array[mid]:
                last = mid - 1
            else:
                first = mid + 1
    return -1

```

* Bubble Sort: worst case ($O(n^2)$), average case ($O(n^2)$), best case ($O(n)$); space ($O(1)$)
* Merge Sort: time ($O(nlog(n))$), space ($O(n)$)
* Quick Sort: generally time ($O(nlog(n))$), space ($O(1)$)

Here's a good explanation of Quick Sort and corresponding Python implementation.

### Maps and Hashing

When we're talking about hash tables, we can define a "load factor":

Load Factor = Number of Entries / Number of Buckets

The purpose of a load factor is to give us a sense of how "full" a hash table is. For example, if we're trying to store 10 values in a hash table with 1000 buckets, the load factor would be 0.01, and the majority of buckets in the table will be empty. We end up wasting memory by having so many empty buckets, so we may want to rehash, or come up with a new hash function with less buckets. We can use our load factor as an indicator for when to rehash—as the load factor approaches 0, the more empty, or sparse, our hash table is. 
On the flip side, the closer our load factor is to 1 (meaning the number of values equals the number of buckets), the better it would be for us to rehash and add more buckets. Any table with a load value greater than 1 is guaranteed to have collisions. 

```python
class HashTable(object):
    def __init__(self):
        self.table = [None]*10000

    def store(self, string):
        hv = self.calculate_hash_value(string)
        if hv != -1:
            if self.table[hv] != None:
                self.table[hv].append(string)
            else:
                self.table[hv] = [string]

    def lookup(self, string):
        hv = self.calculate_hash_value(string)
        if hv != -1:
            if self.table[hv] != None:
                if string in self.table[hv]:
                    return hv
        return -1

    def calculate_hash_value(self, string):
        value = ord(string[0])*100 + ord(string[1])
        return value
```

### Trees

```python
class BinaryTree(object):
    def __init__(self, root):
        self.root = Node(root)

    def search(self, find_val):
        return self.preorder_search(tree.root, find_val)

    def print_tree(self):
        return self.preorder_print(tree.root, "")[:-1]

    def preorder_search(self, start, find_val):
        if start:
            if start.value == find_val:
                return True
            else:
                return self.preorder_search(start.left, find_val) or self.preorder_search(start.right, find_val)
        return False

    def preorder_print(self, start, traversal):
        if start:
            traversal += (str(start.value) + "-")
            traversal = self.preorder_print(start.left, traversal)
            traversal = self.preorder_print(start.right, traversal)
        return traversal
```


```python
# Binary Search Tree
class BST(object):
    def __init__(self, root):
        self.root = Node(root)

    def insert(self, new_val):
        self.insert_helper(self.root, new_val)

    def insert_helper(self, current, new_val):
        if current.value < new_val:
            if current.right:
                self.insert_helper(current.right, new_val)
            else:
                current.right = Node(new_val)
        else:
            if current.left:
                self.insert_helper(current.left, new_val)
            else:
                current.left = Node(new_val)

    def search(self, find_val):
        return self.search_helper(self.root, find_val)

    def search_helper(self, current, find_val):
        if current:
            if current.value == find_val:
                return True
            elif current.value < find_val:
                return self.search_helper(current.right, find_val)
            else:
                return self.search_helper(current.left, find_val)
        return False

```

Heap is a special type of tree that a huge elements are arranged in increasing or decreasing order such that the root element is either the maximum or minimum value in the tree. 

Heaps search: $O(n)$

Delete and insert with heapify: worst case $O(logn)$

Heap is typically stored as a array.

Binary Tree: 
    * Search O(n)
    * Delte O(n)
    * Insert O(log(n))

Binary Search Tree:
    * Search O(log(n))
    * Insert O(log(n))

### Graphs

Disconnected graphs are very similar whether the graph's directed or undirected—there is some vertex or group of vertices that have no connection with the rest of the graph.

A directed graph is weakly connected when only replacing all of the directed edges with undirected edges can cause it to be connected. 

 In a connected graph, there is some path between one vertex and every other vertex.

 Strongly connected directed graphs must have a path from every node and every other node. So, there must be a path from A to B AND B to A.

 Graph representation: vertex and edges objects

 * Edge list
 * Adjacency list
 * Adjacency matrix

```python
# Graph
class Node(object):
    def __init__(self, value):
        self.value = value
        self.edges = []

class Edge(object):
    def __init__(self, value, node_from, node_to):
        self.value = value
        self.node_from = node_from
        self.node_to = node_to

class Graph(object):
    def __init__(self, nodes=[], edges=[]):
        self.nodes = nodes
        self.edges = edges

    def insert_node(self, new_node_val):
        new_node = Node(new_node_val)
        self.nodes.append(new_node)
        
    def insert_edge(self, new_edge_val, node_from_val, node_to_val):
        from_found = None
        to_found = None
        for node in self.nodes:
            if node_from_val == node.value:
                from_found = node
            if node_to_val == node.value:
                to_found = node
        if from_found == None:
            from_found = Node(node_from_val)
            self.nodes.append(from_found)
        if to_found == None:
            to_found = Node(node_to_val)
            self.nodes.append(to_found)
        new_edge = Edge(new_edge_val, from_found, to_found)
        from_found.edges.append(new_edge)
        to_found.edges.append(new_edge)
        self.edges.append(new_edge)

    def get_edge_list(self):
        edge_list = []
        for edge_object in self.edges:
            edge = (edge_object.value, edge_object.node_from.value, edge_object.node_to.value)
            edge_list.append(edge)
        return edge_list

    def get_adjacency_list(self):
        max_index = self.find_max_index()
        adjacency_list = [None] * (max_index + 1)
        for edge_object in self.edges:
            if adjacency_list[edge_object.node_from.value]:
                adjacency_list[edge_object.node_from.value].append((edge_object.node_to.value, edge_object.value))
            else:
                adjacency_list[edge_object.node_from.value] = [(edge_object.node_to.value, edge_object.value)]
        return adjacency_list

    def get_adjacency_matrix(self):
        max_index = self.find_max_index()
        adjacency_matrix = [[0 for i in range(max_index + 1)] for j in range(max_index + 1)]
        for edge_object in self.edges:
            adjacency_matrix[edge_object.node_from.value][edge_object.node_to.value] = edge_object.value
        return adjacency_matrix

    def find_max_index(self):
        max_index = -1
        if len(self.nodes):
            for node in self.nodes:
                if node.value > max_index:
                    max_index = node.value
        return max_index
```

### Case Studies

* Shortest path problem: 
    - Dijkstra's Algorithm for weighted undirected graphs (pick the lowest weight) $O(V^2)$
* Knapsack problem:
    - Brute Force: $O(2^n)$
    - Dynamic Programming
* Traveling Salesman Problem (NP-hard)
    - NP-hard problems don't have a known algorithm that can solve them in polynomial time. 
    - 

### Technical Interview Techniques

* Clarifying the question 
* Generating inputs and output
* Generating test cases
* Brainstorming
* Runtime analysis
* Coding
* Debugging

**Websites:**

* [HackerRank](https://www.hackerrank.com/): Website and community with programming challeges that you can go through for additional practice.

* [Project Euler](https://projecteuler.net/): This website has a ton of logic problems that you can practice writing coded solutions to.

* [Interview Cake](https://www.interviewcake.com/): Practice questions and some tutorials available.

* [Interactive Python](http://interactivepython.org/runestone/static/pythonds/index.html): Loads of tutorials on pretty much every topic covered here and many more, along with Python examples and concept questions.

* [Topcoder](https://www.topcoder.com/): New practice problems every day, and some tech companies use answers to those problems to find new potential hires.

* [LeetCode](https://leetcode.com/): Practice problems, mock interviews, and articles about problems.

* [BigO Cheat Sheet](http://bigocheatsheet.com/): Summary of efficiencies for most common algorithms and data structures.

* [Five Essential Phone Screen Questions](https://sites.google.com/site/steveyegge2/five-essential-phone-screen-questions)



References:

* [Problem solving with Algorithms and Data Structures](http://interactivepython.org/runestone/static/pythonds/index.html)


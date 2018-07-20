---
layout: post
title: Notes for Python
tags: DataScience Python
---

# Notes for Python 

## 1. Environment setting of different python versions

As we know, Python has different version (2.x vs 3.x). How to use two different versions while not mess them up? Using Conda, we can do this. 


** Install a different version of Python **

```python
# Create a 'python3' environment and install Python3
conda create --name python3 python=3

# Activate 'python3' environment
source active python3

# Deactive 'python3' environment
source deactive python3

# Check all environments under Conda
conda info --envs

# Install packages under current environment
conda install package_name
# Or
conda install --name environment_name package_name

# Update certain package
conda update package_name
```







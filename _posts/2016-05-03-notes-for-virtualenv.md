---
layout: post
title: Notes for virtualenv
tags: Python virtualenv
---

# Notes for [virtualenv](https://virtualenv.pypa.io/en/latest/index.html)

Virtualenv is a tool for creating isolated 'virtual' python environment. It makes sure different versions of packages and dependencies can exist in a same machine.

To install:  `pip install virtualenv`

Basic command:

* `virtualenv ENV`: ENV is a directory to place the new virtual enviornment. 
    - `ENV/lib/` and `ENV/include/` are created and used to store library files for a new virtualenv python.
    - `ENV/bin` is created for executables files. Runing a script with `#! /path/to/ENV/bin/python` would run that script under this virtualenv's python.
* Enable a virtual environment: `source ENV/bin/activate`
* Disable a virtual environment: `deactivate`
* Delte a virtual environment: `deactivate` and `rm -r ENV`
* Make envrionment relocatable: `virtualenv --relocatable ENV`

* Check what is installed: 
    - `pip install yolk`
    - `yolk -l`

[Here](http://www.simononsoftware.com/virtualenv-tutorial/) is a great tutorial for `virtualenv`.


To use virtualenv install Python3, we can do the following ([reference](https://stackoverflow.com/questions/23842713/using-python-3-in-virtualenv)):
* `pip install --upgrade virtualenv`
* `virtualenv -p python3 envname`

---
layout: post
title: NetCDF notes
tags: NetCDF
---


* Extract variable “SST” from in.nc 
`$ ncks -v SST in.nc out.nc`
* Delete variable “lev” from in.nc 
`$ ncks -C -O -x -v lev in.nc out.nc`
* Delete dimension “lev” from in.nc 
`$ ncwa -a lev in.nc out.nc`
* Repeack the out.nc after averaging-out the level dimension with “ncwa”
`$ ncpdq in.nc out.nc`

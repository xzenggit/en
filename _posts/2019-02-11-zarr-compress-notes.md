---
layout: post
title: Zarr file compress notes
tags: zarr
---


```python
store = s3fs.S3Map(root=s3_path+zarr_name, s3=s3, check=False)
compressor = zarr.Blosc(cname='zstd', clevel=5)
encoding = {vname: {'compressor': compressor} for vname in tmp_ds.data_vars}
#Try the consolidated=True option in next released version.
tmp_ds.to_zarr(store=store, encoding = encoding, consolidated=True)
```

The compress level `clevel` really affects the file saving speed and size. I did some experiments, and found `clevel=5` is usually a good choice.

Experiment results:
* clevel=9; 18min; 7.6G
* clevel=5; 7min; 8.3G
* clevel=3; 8min; 9.1G
* clevel=0; 8min; 12G

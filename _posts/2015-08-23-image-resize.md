---
layout: post
title: Resize image with ImageMagick
tags: Resize ImageMagick
---

Recently, I need to resize pictures in a certain size with locked ratio. Because this is locked ratio, sometimes I need to add some white space to the pictures. ImageMagick is really a great tool to do this. Here's the script how it works.

```
#!/bin/bash
# This is a bash script.
# Use ImageMagick to resize a picture to certain size with locked ratio.
# The margins will be filled with white color in default. 
# You can also use `-backgroud` parameter to specify the filled color.
# Here, 900x650 is resized picture size in pixels. 
# `-gravity` specifies the position of origional picture in resized picture.
# `-extent` is used to filled the margins in resized pictures. 
# `-tim` is used to trim the white edges of the picture

fname= my_fig.png
convert $fname -trim -resize 900x650 -gravity center -extent 900x650 $fname
```


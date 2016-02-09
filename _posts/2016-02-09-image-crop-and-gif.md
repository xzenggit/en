---
layout: post
title: Notes Image crop and gif animation
tags: image
---
Script for crop png files:

```shell
#!/bin/bash
# crop white space for images in a directory
for X in *.png
do
    echo $X
    convert -trim $X $X
done
exit
# eof
```

Script for making a gif animation:

```shell
#!/bin/bash
# convert png to jpg, and then make a gif with delay 10
mogrify -format jpg *.png
convert -delay 10 -loop 0 *.jpg my_animation.gif
```

Note: to use the `convert` command, you need to install [ImageMagic](http://www.imagemagick.org/script/index.php).

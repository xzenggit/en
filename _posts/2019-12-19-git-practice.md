---
layout: post
title: Git best practice
tags: git
---

When working in git, it's alway safe to have your own branch and do your personal work within it. After everything is ready, then you can merge it with the 
master branch. To do that, we need

```bash
#Create the branch on your local machine and switch in this branch :
$ git checkout -b [name_of_your_new_branch]
#Change working branch :
$ git checkout [name_of_your_new_branch]
#Push the branch on github :
$ git push origin [name_of_your_new_branch]
#You can see all branches created by using :
$ git branch
```

To merge your good code with the master branch, we can (assume test is your own branch)

```bash
$git checkout master
$git pull origin master
$git merge test
$git push origin master
```

References:
* [stackoverflow](https://stackoverflow.com/questions/5601931/what-is-the-best-and-safest-way-to-merge-a-git-branch-into-master)
* [Kunena-Forum](https://github.com/Kunena/Kunena-Forum/wiki/Create-a-new-branch-with-git-and-manage-branches)

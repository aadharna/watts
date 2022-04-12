# Watts
PINSKY v2

This is a rewrite and extension of the UntouchableThunder repo.

Watts is a built to explore Open-Ended Learning algorithms. These are algorithms where the training and evaluation distributions are dynamically changed in response to agent feedback. 
The method of distribution shifting that Watts is exploring (first) is using evolutionary algorithms has an outer meta-learning loop
on top of an inner-loop of agent learning. 

In PINSKY (v1.0), I manually created futures and collected answers after each distributed call.
That version is not scalable properly and also didn't have proper tests. 
The primary goals of version 2.0 is to better use parallelism (e.g. futures), have classes that can be easily extended 
by a user, and to cleanly scale to arbitrary compute. 

----  

Installation:  

    * conda create -n NAME python=3.8
    * conda activate NAME
    * Install pytorch according to your system and environment from here: https://pytorch.org/get-started/locally/
    * If you want to render/record videos of griddly games, you need to install Vulkan. Instructions can be found here: https://github.com/Bam4d/Griddly
    * At the root of this project, run: `pip install -e .`
    * conda install -c conda-forge ffmpeg # remove this with better docker foo

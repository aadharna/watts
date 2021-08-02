# enigma
PINSKY v2

This is a rewrite and extension of the UntouchableThunder repo.

**Furthermore, we aim for Enigma to generically tackle the 
problem of simultaneous learning with generators and solvers.**

In PINSKY (v1.0), I manually created futures and collected answers after each distributed call.
That version is not scalable properly and also didn't have proper tests. 
The primary goals of version 2.0 is to better use parallelism (e.g. futures), have classes that can be easily extended 
by a user, and to cleanly scale to arbitrary compute. 

----  

Installation:  

	* conda create -n NAME python=3.7.10  
	* conda activate NAME  
	* pip install ray  
	* pip install ray[rllib]  
	* conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge  
		* Note, you should grab the correct install from here: https://pytorch.org/get-started/locally/
	* pip install griddly
		* Make sure that vulkan is installed since Griddly depends on Vulkan for its rendering.
    * pip install networkx

If you see stuff like:
```
(pid=41878) Unknown encoder 'libx264'
(pid=41878) ERROR: VideoRecorder encoder exited with status 1
(pid=41872) ERROR: VideoRecorder encoder exited with status 1
(pid=41872) Unknown encoder 'libx264'
(pid=41878) Unknown encoder 'libx264'
(pid=41872) Unknown encoder 'libx264'
(pid=41872) ERROR: VideoRecorder encoder exited with status 1
(pid=41878) ERROR: VideoRecorder encoder exited with status 1
```
you can ignore it. Aaron will fix it later.
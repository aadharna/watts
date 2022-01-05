# watts
PINSKY v2

This is a rewrite and extension of the UntouchableThunder repo.

**Furthermore, we aim for Watts to generically tackle the
problem of simultaneous learning with generators and solvers.**

In PINSKY (v1.0), I manually created futures and collected answers after each distributed call.
That version is not scalable properly and also didn't have proper tests. 
The primary goals of version 2.0 is to better use parallelism (e.g. futures), have classes that can be easily extended 
by a user, and to cleanly scale to arbitrary compute. 

----  

Installation:  

	* make
	* make run

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

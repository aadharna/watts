# enigma
PINSKY v2

This is a rewrite of the UntouchableThunder repo. 

The primary goals of version 2.0 is to better use parallelism (e.g. futures), have classes that can be easily redefined 
by a user, and to cleanly scale to arbitrary compute. 
In PINSKY (v1.0), I manually created futures and collected answers after each distributed call.
That version is not scalable properly and also didn't have proper tests. 

----  


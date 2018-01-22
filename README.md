# Multi-agent Environment

The open-ai multiagent-particle-envs uses continuous outputs. Since the output
parameters are acceleration, the objects fly very far away from the center. For
faster learning, the following changes :

* Grid-world, discretized movement
* Toroidal world, finite-world size


The domains created as of now are :
 
* Simple\_reference 
* Simple\_chaser
* Simple\_cover
* Simple\_speaker\_listener

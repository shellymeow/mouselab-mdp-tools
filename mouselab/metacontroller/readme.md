# Metacontroller Documentation

This folder contains the implementation of the updated hierarchical strategy discovery algorithm. 
This folder originally comes from. [https://github.com/RationalityEnhancement/SSD_Hierarchical/tree/master/Metacontroller](https://github.com/RationalityEnhancement/SSD_Hierarchical/tree/master/Metacontroller)

## How to run BMPS

The easiest way to run BMPS on a new environment is to use the ```optimize``` function from ```metacontroller.py```, which optimizes the VOC weights with Bayesian Optimization. An example of how to do this can be found in ```comparison_meta_hier.ipynb``` in which the goal-switching variant as well as a lessened, purely hierarchical variant of our algorithm are trained and evaluated. 

## Extensions to the mouselab environment
The original mouselab environment has been extended in ``` mouselab_env.py ``` to include the following adjustments: 
1. Computational speedup through tree contraction.
2. An optional adjusted cost function returning the number of nodes needed to compute VPI and VPI_action features. 
3. Updated functions to compute paths to a node while taking the possibility of multiple paths leading to the same node into account. 
4. The option to define the environment using a Normal distribution instead of a Categorical distribution. Behind the scenes the Normal distribution will be binned and treated as a Categorical distribution.
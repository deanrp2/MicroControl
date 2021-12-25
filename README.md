# MircoControl

Repository to house working scripts for optimization problems involving 
HOLOSGen reactor.

## USER INFORMATION
Repository released publicly so users can reproduce results published in 
"Multiobjective Optimization of Nuclear Microreactor Control System Operation 
with Surrogate-based Evolutionary Algorithms" in Annals of Nuclear Energy. 
References to this journal paper are used in this documentation.

## REACTOR INFO:
* Startup reactivity = 3308 pcm

## COORDINATE NOTES:
* quadrant numbering starts with the top-right quadrant and proceeds
counterclockwise
* drum numbering starts from the bottom drum in  quadrant 1 and proceeds counterclockwise 
around each of the drums
* the positive rotation direction is counterclockwise for drums in quadrant 1 and 3
* the positive rotation direction is clockwise for drums in quadrant 2 and 4

## CORE DEMO PROBLEMS:
* Problem 5 (CSA): One drum broken, hit even split and criticality, minimum maximum travel 
distance
* Problem 7 (CSB): Max drum differential worth, hit critical, 0.5% bias in Q1 power

## WEIGHT OPTIMA HISTOGRAM GENERATION
In order to investigate the quality of weight sets for scalarization, multiple independent
optimization routines can be run with a set of weights. Then, the resulting objective
distributions can be analyzed. In order to automatically run these independent optimization
routines, p5_workspace/es_hist.py or p7_workspace/es_hist.py can be run with weights specified
in the `wts` variable. Results will be logged in the file given in the `histname` variable. To
visualize the results, the hist_plot.py file is given. The outputs should be specified in the
`fnames` variable. The datasets used to generate the histograms shown in the paper are included
in the repository in the log directory.

## ALGORITHM PARAMETER OPTIMIZATION
In order to see the results from the parameter optimization routine used in the manuscript, 
scripts are included for each of the  algorithms in p5_workspace and p7_worspace with the 
*_tune.py name. 

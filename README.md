# Mircoreactor Control Optimization

Repository to house working scripts for optimization problems involving 
HOLOSGen reactor.

## USER INFORMATION
Repository released publicly so users can reproduce results published in 
"Multiobjective Optimization of Nuclear Microreactor Control System Operation 
with Surrogate-based Evolutionary Algorithms" in Nuclear Engineering and Design. 
References to this journal paper are used in this documentation.

User must have NEORL installed:
https://neorl.readthedocs.io/en/latest/

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

# QUICK TESTING
Within the p5_workspace and p7_workspace directories, there are 6 files which contain the *_expl.py 
suffix. Each of these can be run for a single optimization routine performed on either problem 5
or problem 7. To get the final optimization runs used in verification with Serpent, run either
p5_workspace/mfo_final.py or p7_workspace/pso_final.py.

For example, to run the differential evolution algorithm for problem 5, navigate to p5_workspace and 
run:

`python3 de_expl.py`

# DETAILED TESTING
With these directions, you are able to reporoduce the results shown in the journal paper.

## WEIGHT OPTIMA HISTOGRAM GENERATION
In order to investigate the quality of weight sets for scalarization, multiple independent
optimization routines can be run with a set of weights. Then, the resulting objective
distributions can be analyzed. In order to automatically run these independent optimization
routines, p5_workspace/es_hist.py or p7_workspace/es_hist.py can be run with weights specified
in the `wts` variable. Results will be logged in the file given in the `histname` variable. To
visualize the results, the hist_plot.py file is given. The outputs should be specified in the
`fnames` variable. The datasets used to generate the histograms shown in the paper are included
in the repository in the log directory. Generating these histograms is computationally expensive,
the script can be terminated when enough samples are written to the output file.

## ALGORITHM PARAMETER OPTIMIZATION
In order to see the results from the parameter optimization routine used in the manuscript, 
scripts are included for each of the  algorithms in p5_workspace and p7_worspace with the 
*_tune.py name. In order to rerun these optimizations, the `tune` variable can be set to `True`.
This will overwrite the existing data files giving the results from the grid search.

## ALGORITHM POPULATION STATISTICS
In order to generate figures like the ones seen in Figure 7 and Figure 9 in the journal paper, 
a script is given in p5_workspace and p7_workspace named green_plots.py.

## ALGORITHM OBJECTIVE EVOLUTION
In order to generate figures like the ones seen in Figure 8 and Figure 10 in the journal paper,
scripts are given for each of the six algorithms in p5_workspace and p7_workspace named 
*_expl.py. The number of function evaluations used in the optimization routine is specified in 
the function call under the `if __name__ == "__main__":` line in the argument of *_expl(\#\#).

## ALGORITHM PERFORMANCE SUMMARY STATISTICS
To generate statistics which characterize the optima achieved by the various optimization 
algorithms, a script named results_table.py is provided in p5_workspace and p7_workspace. 
The results will be written to a file called results_table.txt. The existing results_table.txt
file contains the statistics that are published in Table 5 and 6 of the journal paper. It can
be very computationally expensive to generate these results so the user can adjust 3 different
variables to make the results faster. First, `fevals` can be changed to reduce the number of 
function evaluations done in each optimization routine. Second, `runs` can be reuced to reduce 
the number of runs used to generate the summary statistics. Third, the `places` variable can 
be used to only select certain algorithms to run. Currently, those parameters are set to the 
values used in the paper.












# MircoControl

Repository to house working scripts for optimization problems involving 
HOLOSGen reactor.

REACTOR INFO:
* All drums in k=0.967987
* Startup reactivity = 3072 pcm
* beta_eff = 0.0148166

COORDINATE NOTES:
* quadrant numbering starts with the top-right quadrant and proceeds
counterclockwise
* drum numbering starts from the bottom drum in  quadrant 1 and
proceeds counterclockwise around each of the drums
* the positive rotation direction is counterclockwise for drums in quadrant 1 and 3
* the positive rotation direction is clockwise for drums in quadrant 2 and 4

DEMO PROBLEMS:
1. Rotate 8 drums in unison to achieve some desired reactivity
2. Rotate 8 drums independently to maximumize/minimize criticality
3. From fully inserted, minimum travel distance to critical, even power split
    * can be used to tell if drums should be given different motors?
4. Find config with 0.5% bias in Q1 power, criticality and minimum maximum travel dist
5. One drum broken, hit even split and criticality, minimum maximum tracel dist
6. max drum differential worth, hit critical, even power split
    * theoretical best operating condition, startup time not limited by drum rotation speed
7. max drum differential worth, hit critical, 0.5% bias in Q1 power
    * theoretical best operating condition, startup time not limited by drum rotation speed
8. no constraints, maximum differential worth

CORE DEMO PROBLEMS:\
Problem 5 and 7 were selected to be the main problems used to demonstrate the capabilities of any
optimization algoritms.
As such, they are explored further.

OBJECTIVE WEIGHTS:\
5. \
   Reactivity Error  - 0.6\
   Power Split Error - 0.2\
   Travel Distance   - 0.1\
7. \
   Reactivity Error  - 0.5\
   Power Split Error - 0.3\
   Diff. Worth Max.  - 0.2\



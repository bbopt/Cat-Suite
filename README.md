# Cat-Suite: a collection of optimization problems with categorical and quantitative variables for benchmarking
*Under construction!*

This repository contains 60 mixed-variable analytical problems with categorical and quantitative variables for benchmarking. Half of the problems are constrained. 

## Summary and problem characteristics
The following two tables present the characteristics of the unconstrained and constrained problems, respectively. The star symbol represents a user-chosen parameter. 


## Citation

If you use this benchmark collection in your work, please cite:

Hallé-Hannan, E., Audet, C., Diouane, Y., Le Digabel, S., and Tribes, C.
Cat-Suite: A collection of optimization problems with categorical and quantitative variables for benchmarking.
Technical Report G-2025-29, Les cahiers du GERAD, 2025.
Available at: https://www.gerad.ca/fr/papers/G-2025-39



**Table 1:** Unconstrained test problems.

| Name | $n^{cat}$ | $\ell$ | $n^{int}$ | $n^{cont}$ | Smooth | Ref. | Original problem |
|------|-------------|--------|----------------|-------------------|--------|------|------------------|
| Cat-1  | 2 | 9 | 2 | $\star$ | No | [4] | Ackley |
| Cat-2  | 2 | 9 | 2 | 3 | No | [4] | Beale |
| Cat-3  | 2 | 4 | 2 | 2 | Yes | [7] | Augmented-Branin |
| Cat-4  | 2 | 4 | 2 | 4 | No | [4] | Bukin-6 |
| Cat-5  | 1 | 6 | 1 | 3 | Yes | [5] | EVD-52 |
| Cat-6  | 2 | 9 | 0 | 2 | Yes | [7] | Goldstein |
| Cat-7  | 3 | 16 | 3 | 2 | No | [4] | Goldstein-Price |
| Cat-8  | 1 | 4 | 1 | 5 | Yes | [5] | HS78 |
| Cat-9  | 2 | 9 | 2 | $\star$ | No | [4] | Rastrigin |
| Cat-10 | 2 | 6 | 2 | $\star$ | No | [4] | Rosenbrock |
| Cat-11 | 1 | 4 | 1 | 4 | Yes | [5] | Rosen-Suzuki |
| Cat-12 | 1 | 5 | $\star$ | $\star$ | No | [4] | Styblinski-Tang |
| Cat-13 | 1 | 10 | 0 | 4 | Yes | [6] | Toy |
| Cat-14 | 1 | 10 | 0 | 8 | No | [6] | Toy |
| Cat-15 | 1 | 5 | 3 | 4 | Yes | [5] | Wong-1 |
| Cat-16 | 2 | 9 | 2 | $\star$ | No | [4] | Zakharov |
| Cat-17 | 2 | 49 | 1 | 7 | No | [9] | Ishigami |
| Cat-18 | 2 | 100 | 1 | 7 | No | [9] | Hartmann |
| Cat-19 | 2 | 64 | 0 | $\star$ | No | [9] | Levy |
| Cat-20 | 3 | 80 | 2 | 4 | No | [9] | Camel |
| Cat-21 | 1 | 61 | 0 | 4 | No | [5] | Gamma |
| Cat-22 | 1 | 51 | 0 | 6 | Yes | [5] | EVD-61 |
| Cat-23 | 3 | 12 | 0 | 5 | No | [3] | Hal-04 |
| Cat-24 | 1 | 21 | 2 | 3 | Yes | [5] | OET5 |
| Cat-25 | 1 | 18 | 10 | 10 | Yes | [5] | Wong 3 |
| Cat-26 | 1 | 10 | 3 | 5 | Yes | [1] | Roustant |
| Cat-27 | 2 | 100 | 1 | 4 | Yes | [1] | Kowalik-Osborne |
| Cat-28 | 2 | 36 | 0 | 2 | Yes | [4] | Three-Hump |
| Cat-29 | 2 | 64 | 3 | 4 | Yes | [4] | McCormick |
| Cat-30 | 4 | 81 | 4 | 6 | Yes | [4] | Shekel |


**Table 2:** Constrained test problems.

| Name | $n^{cat}$ | $\ell$ | $n^{int}$ | $n^{cont}$ | $m$ | Smooth | Ref. | Original problem |
|------|-------------|--------|----------------|-------------------|-----|--------|------|------------------|
| Cat-cstrs-1 | 2 | 9 | 2 | 3 | 3 | No | [4] | Beale |
| Cat-cstrs-2 | 2 | 4 | 2 | 2 | 1 | Yes | [7] | Augmented-Branin |
| Cat-cstrs-3 | 2 | 9 | 2 | 4 | 2 | No | [4] | Bukin-6 |
| Cat-cstrs-4 | 1 | 4 | 4 | 4 | 3 | Yes | [5] | Dembo-5 |
| Cat-cstrs-5 | 1 | 6 | 1 | 3 | 1 | No | [5] | EVD-52 |
| Cat-cstrs-6 | 2 | 9 | 2 | 3 | 4 | Yes | [1] | G-09 |
| Cat-cstrs-7 | 2 | 9 | 0 | 2 | 1 | Yes | [7] | Goldstein |
| Cat-cstrs-8 | 2 | 25 | 2 | 2 | 2 | Yes | [4] | Himmelblau |
| Cat-cstrs-9 | 2 | 4 | 3 | 5 | 4 | Yes | [5] | HS-114 |
| Cat-cstrs-10 | 1 | 3 | 2 | 4 | 6 | Yes | [5] | Pentagon |
| Cat-cstrs-11 | 1 | 8 | 2 | 2 | 3 | Yes | [1] | Pressure-Vessel |
| Cat-cstrs-12 | 2 | 25 | 1 | 2 | 2 | Yes | [1] | Reinforced-Concrete |
| Cat-cstrs-13 | 2 | 6 | 2 | $\star$ | 1 | No | [4] | Rosenbrock |
| Cat-cstrs-14 | 1 | 5 | $\star$ | $\star$ | 2 | No | [4] | Styblinski–Tang |
| Cat-cstrs-15 | 1 | 10 | 0 | 4 | 2 | Yes | [6] | Toy |
| Cat-cstrs-16 | 1 | 6 | 4 | 6 | 3 | Yes | [5] | Wong-2 |
| Cat-cstrs-17 | 2 | 64 | 0 | 6 | 11 | No | [1] | Speed-reducer |
| Cat-cstrs-18 | 1 | 50 | 2 | 2 | 5 | No | [1] | Spring |
| Cat-cstrs-19 | 6 | 64 | 2 | 2 | 8 | No | [1] | G07 |
| Cat-cstrs-20 | 2 | 100 | 2 | 7 | 10 | Yes | [1] | Car-side-impact |
| Cat-cstrs-21 | 1 | 18 | 0 | 16 | 7 | No | [5] | Dembo-7 |
| Cat-cstrs-22 | 1 | 12 | 0 | 12 | 3 | No | [5] | MAD |
| Cat-cstrs-23 | 1 | 13 | 10 | 10 | 4 | No | [5] | Wong 3 |
| Cat-cstrs-24 | 2 | 100 | 2 | 4 | 5 | No | [8] | Welded-beam |
| Cat-cstrs-25 | 1 | 10 | 0 | 2 | 3 | No | [8] | Three-bar truss |
| Cat-cstrs-26 | 2 | 36 | 2 | 2 | 6 | No | [4] | Three-hump |
| Cat-cstrs-27 | 2 | 64 | 3 | 4 | 3 | No | [4] | McCormick |
| Cat-cstrs-28 | 2 | 100 | 2 | 2 | 5 | No | [2] | G06 |
| Cat-cstrs-29 | 4 | 81 | 4 | 6 | 3 | No | [4] | Shekel |
| Cat-cstrs-30 | 2 | 49 | 1 | 7 | 3 | No | [9] | Ishigami |




## Best known feasible values: 13 June 2025

**Table 3:** Best known values for unconstrained problems.

| Problem   | $f(x_{\text{best}})$ |
|-----------|----------------------|
| Cat-1     | 21.71                |
| Cat-2     | 3.12E-12             |
| Cat-3     | 4.87                 |
| Cat-4     | 1.06E4               |
| Cat-5     | -31250.5             |
| Cat-6     | 38.08                |
| Cat-7     | 5                    |
| Cat-8     | -152                 |
| Cat-9     | -2                   |
| Cat-10    | 1.03                 |
| Cat-11    | -113.71              |
| Cat-12    | -102.51              |
| Cat-13    | -0.71                |
| Cat-14    | 0.14                 |
| Cat-15    | -1942.82             |
| Cat-16    | 1                    |

**Table 4:** Best known feasible values for constrained problems.

| Problem       | $f(x_{\text{best}})$ |
|---------------|----------------------|
| Cat-cstrs-1   | 1.27E-03             |
| Cat-cstrs-2   | -5.1273              |
| Cat-cstrs-3   | 4.30E-03             |
| Cat-cstrs-4   | -24245741.22         |
| Cat-cstrs-5   | -77237.8             |
| Cat-cstrs-6   | 555.58               |
| Cat-cstrs-7   | 38.8                 |
| Cat-cstrs-8   | 10                   |
| Cat-cstrs-9   | -1256527.34          |
| Cat-cstrs-10  | 1.47E-09             |
| Cat-cstrs-11  | 6184.75              |
| Cat-cstrs-12  | 303.4                |
| Cat-cstrs-13  | 19210.88             |
| Cat-cstrs-14  | -66.68               |
| Cat-cstrs-15  | 3                    |
| Cat-cstrs-16  | -9721.58             |


## References

[1] A.-S. Crélot, C. Beauthier, D. Orban, C. Sainvitu, and A. Sartenaer.
Combining Surrogate Strategies with MADS for Mixed-Variable Derivative-Free Optimization.
Technical Report G-2017-70, Les cahiers du GERAD, 2017.

[2] N. I. M. Gould, D. Orban, and Ph. L. Toint.
CUTEst: a Constrained and Unconstrained Testing Environment with Safe Threads for Mathematical Optimization.
Computational Optimization and Applications, 60(3):545–557, 2015.

[3] M. Halstrup.
Black-Box Optimization of Mixed Discrete-Continuous Optimization Problems.
PhD thesis, Technical University of Denmark, 2016.

[4] M. Jamil and X.-S. Yang.
A literature survey of benchmark functions for global optimisation problems.
International Journal of Mathematical Modelling and Numerical Optimisation, 4(2):150–194, 2013.

[5] L. Lukšan and J. Vlček.
Test Problems for Nonsmooth Unconstrained and Linearly Constrained Optimization.
Technical Report V-798, ICS AS CR, 2000.

[6] M. Munoz Zuniga and D. Sinoquet.
Global optimization for mixed categorical-continuous variables based on Gaussian process models with a randomized categorical space exploration step.
INFOR: Information Systems and Operational Research, 58(2):310–341, 2020.

[7] J. Pelamatti, L. Brevault, M. Balesdent, E.-G. Talbi, and Y. Guerin.
Efficient global optimization of constrained mixed variable problems.
Journal of Global Optimization, 73(3):583–613, 2019.

[8] T. Ray and K. M. Liew.
A Swarm Metaphor for Multiobjective Design Optimization.
Engineering Optimization, 34(2):141–153, 2002.

[9] S. Surjanovic and D. Bingham.
Virtual Library of Simulation Experiments: Test Functions and Datasets.
Technical Report, Simon Fraser University, 2025.

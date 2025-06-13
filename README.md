# Cat-Suite: Mixed-variable analytical problems with categorical and quantitative variables for benchmarking
*Under construction!*

This repository contains 32 mixed-variable analytical problems with categorical and quantitative variables for benchmarking. Half of the problems are constrained. 
The following two tables present the characteristics of the unconstrained and constrained problems, respectively. The star symbol represents a user-chosen parameter. 

**Table 1:** Unconstrained test problems.

| Name    | $n^{\text{cat}}$ | $\ell$ | $n^{\text{int}}$ | $n^{\text{cont}}$ | Smooth | Ref. | Original problem     |
|---------|------------|--------|------------------|--------------------|--------|------|----------------------|
| Cat-1   | 2          | 9      | 2                | $\star$            | No     | [2]  | Ackley              |
| Cat-2   | 2          | 9      | 2                | 3                  | No     | [2]  | Beale               |
| Cat-3   | 2          | 4      | 2                | 2                  | Yes    | [5]  | Augmented-Branin    |
| Cat-4   | 2          | 4      | 2                | 4                  | No     | [2]  | Bukin-6             |
| Cat-5   | 1          | 6      | 1                | 3                  | Yes    | [3]  | EVD-52              |
| Cat-6   | 2          | 9      | 0                | 2                  | Yes    | [5]  | Goldstein           |
| Cat-7   | 3          | 16     | 3                | 2                  | No     | [2]  | Goldstein-Price     |
| Cat-8   | 1          | 4      | 1                | 5                  | Yes    | [3]  | HS78                |
| Cat-9   | 2          | 9      | 2                | $\star$            | No     | [2]  | Rastrigin           |
| Cat-10  | 2          | 6      | 2                | $\star$            | No     | [2]  | Rosenbrock          |
| Cat-11  | 1          | 4      | 1                | 4                  | Yes    | [3]  | Rosen-Suzuki        |
| Cat-12  | 1          | 5      | $\star$          | $\star$            | No     | [2]  | Styblinski-Tang     |
| Cat-13  | 1          | 10     | 0                | 4                  | Yes    | [4]  | Toy                 |
| Cat-14  | 1          | 10     | 0                | 8                  | No     | [4]  | Toy                 |
| Cat-15  | 1          | 5      | 3                | 4                  | Yes    | [3]  | Wong-1              |
| Cat-16  | 2          | 9      | 2                | $e$                | No     | [2]  | Zakharov            |


**Table 2:** Constrained test problems.

| Name          | $n^{\text{cat}}$ | $\ell$ | $n^{\text{int}}$ | $n^{\text{cont}}$ | $m$ | Smooth | Ref. | Original problem       |
|---------------|------------|--------|------------------|--------------------|-----|--------|------|------------------------|
| Cat-cstrs-1   | 2          | 9      | 2                | 3                  | 3   | No     | [2]  | Beale                 |
| Cat-cstrs-2   | 2          | 4      | 2                | 2                  | 1   | Yes    | [5]  | Augmented-Branin      |
| Cat-cstrs-3   | 2          | 9      | 2                | 4                  | 2   | No     | [2]  | Bukin-6               |
| Cat-cstrs-4   | 1          | 4      | 4                | 4                  | 3   | Yes    | [3]  | Dembo-5               |
| Cat-cstrs-5   | 1          | 6      | 1                | 3                  | 1   | No     | [3]  | EVD-52                |
| Cat-cstrs-6   | 2          | 9      | 2                | 3                  | 4   | Yes    | [1]  | G-09                  |
| Cat-cstrs-7   | 2          | 9      | 0                | 2                  | 1   | Yes    | [5]  | Goldstein             |
| Cat-cstrs-8   | 2          | 25     | 2                | 2                  | 2   | Yes    | [2]  | Himmelblau            |
| Cat-cstrs-9   | 2          | 4      | 3                | 5                  | 4   | Yes    | [3]  | HS-114                |
| Cat-cstrs-10  | 1          | 3      | 2                | 4                  | 6   | Yes    | [3]  | Pentagon              |
| Cat-cstrs-11  | 1          | 8      | 2                | 2                  | 3   | Yes    | [1]  | Pressure-Vessel       |
| Cat-cstrs-12  | 2          | 25     | 1                | 2                  | 2   | Yes    | [1]  | Reinforced-Concrete   |
| Cat-cstrs-13  | 2          | 6      | 2                | $\star$            | 1   | No     | [2]  | Rosenbrock            |
| Cat-cstrs-14  | 1          | 5      | $\star$          | $\star$            | 2   | No     | [2]  | Styblinski–Tang       |
| Cat-cstrs-15  | 1          | 10     | 0                | 4                  | 2   | Yes    | [4]  | Toy                   |
| Cat-cstrs-16  | 1          | 6      | 4                | 6                  | 3   | Yes    | [3]  | Wong-2                |

### References

[1] A.-S. Crélot, C. Beauthier, D. Orban, C. Sainvitu, and A. Sartenaer.  
*Combining Surrogate Strategies with MADS for Mixed-Variable Derivative-Free Optimization.*  
Technical Report G-2017-70, Les cahiers du GERAD, 2017.

[2] M. Jamil and X.-S. Yang.  
*A literature survey of benchmark functions for global optimisation problems.*  
International Journal of Mathematical Modelling and Numerical Optimisation, 4(2):150–194, 2013.

[3] L. Lukšan and J. Vlček.  
*Test Problems for Nonsmooth Unconstrained and Linearly Constrained Optimization.*  
Technical Report V-798, ICS AS CR, 2000.

[4] M. Munoz Zuniga and D. Sinoquet.  
*Global optimization for mixed categorical-continuous variables based on Gaussian process models with a randomized categorical space exploration step.*  
INFOR: Information Systems and Operational Research, 58(2):310–341, 2020.

[5] J. Pelamatti, L. Brevault, M. Balesdent, E.-G. Talbi, and Y. Guerin.  
*Efficient global optimization of constrained mixed variable problems.*  
Journal of Global Optimization, 73(3):583–613, 2019.

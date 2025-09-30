# Code for "On Computing the Copositive Minimum and its Representatives"
This repository contains an implementation of the algorithms described in the paper "On Computing the Copositive Minimum and its Representatives" by Oertel and Schürmann (2025, arxiv link). Moreover, the code as well as the example matrices used in testing and generating the numerical results are provided.

## Requirements
The code was run using Python 3.12.3. Further requirements are Numpy 1.26.4, Gurobi 11.0.3 (for solving convex quadratic problems), cvxpy 1.6.4 (+CVXOPT Solver 1.3.0 and OSQP Solver 0.6.7.post3) and pandas 2.1.4 (for testing). Note that these are only the versions we developed against, but we believe that most current versions should suffice.

## Usage
To calculate the copositive minimum of a strictly copositive matrix use `python src/calculate_cop_min.py <O or SPN> <filename>`.
The argument *O* or *SPN* decides whether the one difficult coordinate code or the SPN code is used.

To run the testing of the example matrices run `python run_tests.py` twice. The first time generates a file *results* in which the results of the second run are recorded. This script supports stopping and resuming by just aborting and restarting, with the intermediate results being recorded in the *results* files.

## Notes on the Code
This is an in large parts unoptimized and preliminary implementation. There is some support for using arbitrary simplicial cones, instead of the nonnegative orthant, as well (in the one negative difficult coordinate case). There is, although a strategy is indicated in the paper, currently no support for matrices, where no principle rearrangement admits an LDLT-decomposition, since these matrices seem to be quite rare.

A more general and more polished code is planned.

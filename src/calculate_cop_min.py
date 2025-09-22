import time
tic = time.perf_counter()

import numpy as np
import sys

import ldl

import cvxpy as cp
import gurobipy as gp

def navigation_step(Q, j, lamb, recalculate_bounds_lamb, last_reset):
    #Calculate Bounds if necessary and set x appropriatly
    if j <= last_reset:
        Q.update_bounds(j, lamb)
        Q.reset_coord(j)
        last_reset = j - 1
        recalculate_bounds_lamb[j - 1] = False

    elif recalculate_bounds_lamb[j - 1] == True:
        Q.update_bounds(j, lamb)
        #current coordinate might now be smaller than lower bound
        Q.match_coord_to_bound(j)
        recalculate_bounds_lamb[j - 1] = False

    if Q.coord_is_in_bounds(j):
        j -= 1

    else:
        if j == Q.n:
            return j + 1, 0 #exit condition
        else:
            Q.go_next_coord(j + 1)
            #every variable not fixed from j on needs new bounds
            last_reset = j
            j += 1

    return j, last_reset

def calculation_step(Q, j, lamb, recalculate_bounds_lamb, L):
    Q.update_bounds(1, lamb)
    lower, upper = Q.get_bounds_1()
    Q.reset_coord(1)

    #discard zero vector
    if Q.zero_coordinates():
        Q.go_next_coord(1)

    while Q.coord_1() <= upper:
        value = Q.evaluate()

        #if bounds are correctly computed, this is impossible:
        if value > lamb:
            print("???")
            print(value, Q.copy_of_current_coord(), lamb)
            print(Q.get_bounds_1(), Q.coord_1())

        #next short vector (of previos known shortest length)
        elif value == lamb:
            L.append(Q.copy_of_current_coord())

        else: #new shortest vector
            L[:] = [Q.copy_of_current_coord()]
            lamb = value

            if value <= 0:
                raise ValueError(f"Not strictly copositve! {Q.copy_of_current_coord()} (a permutation, resp.) has Q[x] <= 0")

            #new bounds and possibly later starting point
            Q.update_bounds(1, lamb)
            lower, upper = Q.get_bounds_1()
            Q.match_coord_to_bound(1)

            #all variables need new bounds
            recalculate_bounds_lamb[:] = [True] * Q.n

        Q.go_next_coord(1)

    #next branch
    Q.go_next_coord(2)
    j += 1

    return j, lamb


def main():
    if len(sys.argv) < 3:
        print("USAGE: algorithm_type (\"O\" or 0 for one difficult coordinate code, \"SPN\" or 1 for SPN code) file_name")
        sys.exit()
    algo_type = sys.argv[1]
    if algo_type in ["O", "0"]:
        import form_one_diff as form
    elif algo_type in ["SPN", "1"]:
        import form_spn as form
    else:
        print("USAGE: algorithm_type (\"O\" or 0 for one difficult coordinate code, \"SPN\" or 1 for SPN code) file_name")
        sys.exit()

    #start the timing
    tac = time.perf_counter()

    file_name = sys.argv[2]
    Q = form.Form(file_name)
    n = Q.n

    recalculate_bounds_lamb = [False] * n
    last_reset = n

    j = n
    lamb = min([Q.Q[i,i] for i in range(0,n)]) #smallest diagonal element
    L = [] #List of small vectors

    while True:

        if j == 1: #leafs of the search tree
            j, lamb = calculation_step(Q, j, lamb, recalculate_bounds_lamb, L)

        else:
            j, last_reset = navigation_step(Q, j, lamb, recalculate_bounds_lamb, last_reset)

            if j == n + 1: #finish condition
                break

    #retransformation
    for v in L:
        v_np = np.array(v)
        print((Q.C @ v_np).astype(int))
    toc = time.perf_counter()

    print(f"in {toc - tic}s insgesamt, {toc - tac} in main, insgesamt {len(L)} kurze Vektoren")


if __name__ == '__main__':
    sys.exit(main())

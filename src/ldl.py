import numpy as np
from fractions import Fraction


def find_index_of_big_diagonal_element(A):
    all_zero = True
    n = A.shape[0]
    i = 0
    for (j, elem) in enumerate([A[k,k] for k in range(0, n)]):
        if all_zero and elem != 0:
            all_zero = False
            i = j
            continue
        if elem > A[i,i] and elem != 0:
            i = j
    return i


#number_fixed says how many coordinates have to remain fixed (so small diagonal elements under the first elements only)
def find_index_of_small_diagonal_element_fixed(A, number_fixed):
    i = 0
    n = A.shape[0]
    for (j, elem) in enumerate([A[k,k] for k in range(0, n - number_fixed)]):
        if (elem < A[i,i] and elem > 0) or (A[i,i] <= 0 and elem > 0):
            i = j
    return i


def LDL_pivot_big_first(Q):
    n = Q.shape[0]
    perm = [i for i in range(0, n)]
    L = np.zeros((n,n), dtype = Fraction)
    diagonal_elements = []

    A = np.copy(Q)
    for k in range(0, n - 1):
        #done if A == 0
        if not np.any(A):
            diagonal_elements.append(0)
            continue


        i = find_index_of_big_diagonal_element(A)
        #store permutation
        perm[k], perm[i + k] = perm[i + k], perm[k]
        #change rows and columns and update L
        helper = np.copy(A[0, :])
        A[0, :] = A[i, :]
        A[i, :] = helper

        helper = np.copy(A[:, i])
        A[:, i] = A[:, 0]
        A[:, 0] = helper

        helper = np.copy(L[i + k, :])
        L[i + k, :] = L[k, :]
        L[k, :] = helper

        E = A[0:1, 0:1]
        C = A[1:, 0]
        B = A[1:, 1:]

        if E[0,0] == 0:
            print("unimplemented special case: Matrix is not pertrid")
            sys.exit()

        L[(k + 1):, k] = 1/E[0,0] * C
        diagonal_elements.append(E[0,0])
        A_old = A[:,:]

        A = B - 1/E[0,0] * np.atleast_2d(C).T @ np.atleast_2d(C)

    diagonal_elements.append(A[0,0])

    return L, diagonal_elements, perm

def LDL_pivot_small_first(Q, number_fixed):
    n = Q.shape[0]
    perm = [i for i in range(0, n)]
    L = np.zeros((n,n), dtype = Fraction)
    diagonal_elements = []

    A = np.copy(Q)

    for k in range(0, n - 1):
        if not np.any(A):
            diagonal_elements.append(0)
            continue

        i = find_index_of_small_diagonal_element_fixed(A, number_fixed)
        #store permutation
        perm[k], perm[i + k] = perm[i + k], perm[k]
        #change rows and columns and update L
        helper = np.copy(A[0, :])
        A[0, :] = A[i, :]
        A[i, :] = helper

        helper = np.copy(A[:, i])
        A[:, i] = A[:, 0]
        A[:, 0] = helper

        helper = np.copy(L[i + k, :])
        L[i + k, :] = L[k, :]
        L[k, :] = helper

        E = A[0:1, 0:1]
        C = A[1:, 0]
        B = A[1:, 1:]

        if E[0,0] == 0:
            print("unimplemented special case: Matrix is not pertrid")
            sys.exit()

        L[(k + 1):, k] = 1/E[0,0] * C
        diagonal_elements.append(E[0,0])
        A_old = A[:,:]
        A = B - 1/E[0,0] * np.atleast_2d(C).T @ np.atleast_2d(C)

    diagonal_elements.append(A[0,0])

    return L, diagonal_elements, perm



def LDL_without_pivot(Q):
    n = Q.shape[0]
    L = np.zeros((n,n))
    diagonal_elements = []

    A = Q[:, :]

    for k in range(0, n - 1):
        if not np.any(A):
            diagonal_elements.append(0)
            continue

        E = A[0:1, 0:1]
        C = A[1:, 0]
        B = A[1:, 1:]

        L[(k + 1):, k] = 1/E[0,0] * C
        diagonal_elements.append(E[0,0])
        A_old = A[:,:]
        A = B - 1/E[0,0] * np.atleast_2d(C).T @ np.atleast_2d(C)

    diagonal_elements.append(A[0,0])

    return L, diagonal_elements

def LDL_procedure(Q):
    n = Q.shape[0]
    L_old, diag_old, perm_old = LDL_pivot_big_first(Q)
    P_old = np.eye(n, dtype = int)[perm_old].T
    Q_intermediate = (Q[perm_old].T)[perm_old]

    k = 0
    while k < n and diag_old[k] > 0:
        k += 1

    R = np.copy(Q_intermediate).astype(Fraction)
    if k == n - 1:
        R[-1, -1] = R[-1, -1] - diag_old[-1]
    else:
        R_diag_help = diag_old[:]
        R_diag_help[:k] = np.array([0] * k)
        L_help = L_old + np.eye(n, dtype=Fraction)
        R -= L_help @ np.diag(R_diag_help) @ L_help.T

    L_, diag, perm = LDL_pivot_small_first(R, n - k)
    L_aea = L_ + np.eye(n, dtype = int)

    P_help = np.eye(n, dtype = int)[perm].T
    P = P_old @ P_help

    if k <= n - 1:
        diag[-(n-k):] = diag_old[-(n-k):]
        L_[-(n-k):, -(n-k)] = L_old[-(n-k):, -(n-k)]

    L = L_ + np.eye(n, dtype = Fraction)
    return L, diag, P

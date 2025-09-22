import numpy as np
from fractions import Fraction
import sys
import ldl
import cvxpy as cp
import cvxopt


def read_matrix(file_name):
    with open(file_name, "r") as file:
        lines = file.read().splitlines()

        #Dimensions of the matrix
        dimensions = lines[0].split(" ")
        n, m = int(dimensions[0]), int(dimensions[1])
        if n != m:
            raise ValueError("Matrix is not square")

        lines.pop(0)

        matrix_help = []
        for line in lines:
            row_entries = [entry for entry in line.split(" ") if entry != '']
            matrix_help.append([])
            number_elements_in_row = 0
            for entry in row_entries:
                number_elements_in_row += 1
                if "/" in entry:
                    numerator_str, denomenator_str = entry.split("/")
                    number = Fraction(int(numerator_str), int(denomenator_str))
                else:
                    number = Fraction(int(entry), 1)
                matrix_help[-1].append(number)

            if number_elements_in_row != n:
                raise ValueError(f"In row {len(matrix_help)} wrong number of elements: {number_elements_in_row}, should be {n}")

        return(np.array(matrix_help))

class Coordinate:
    def __init__(self):
        self.current_pos = 0

        self.lower_bound = 0
        self.upper_bound = 0

    def update_bounds(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def update_cone_bounds(self, lower_bound, upper_bound):
        self.cone_lower_bound = lower_bound
        self.cone_upper_bound = upper_bound

    def initialize_trivial_cone_bounds(self, lower_bound, upper_bound):
        self.cone_lower_bound_trivial = lower_bound
        self.cone_upper_bound_trivial = upper_bound

        self.update_cone_bounds(lower_bound, upper_bound)

    def reset_pos(self):
        self.current_pos = max(self.cone_lower_bound, self.lower_bound)

    def go_next(self):
        self.current_pos += 1
        # return self.current_pos <= self.bounds[-1][1]
        #this might be unnecessary?
        return self.current_pos

    def in_bounds(self):
        if (self.current_pos <= min(self.cone_upper_bound, self.upper_bound)
            and self.current_pos >= max(self.cone_lower_bound, self.lower_bound)):
            return True
        return False


class Form:
    def __init__(self, filename):
        #Option to decide how the difficult coordinates get bounded.
        #maximizing does not work very well and could be dropped
        self.maximize_sdp = False

        self.Q_ = read_matrix(filename)
        self.n = self.Q_.shape[0]
        self.coordinates = []
        for i in range(0,self.n):
            self.coordinates.append(Coordinate())
        self.initialize_LLL()
        self.initialize_LDL()
        self.initialize_trivial_cone_bounds()
        self.initialize_PSN_decomposition()

    def initialize_LLL(self):
        #do nothing for now
        self.S = np.identity(self.n)
        self.Q_LLL = np.copy(self.Q_)
        self.LLL_done = False



    def initialize_LDL(self):
        self.L, self.diag, self.P = ldl.LDL_procedure(self.Q_LLL)
        self.U = self.L.transpose()

        #get matrix with rows and columns permutated
        self.Q = self.P.T @ self.Q_LLL @ self.P

        self.C = self.S @ self.P

        #Here the cone is just R^n_>=0


    #initialize the range of the variables based on the cone
    def initialize_trivial_cone_bounds(self):
        #R^n_>=0

        for j in range(0, self.n):
            self.coordinates[j].initialize_trivial_cone_bounds(0, np.inf)

    def initialize_PSN_decomposition(self):

        #find how many bounds are difficult
        k = 0
        while k < self.n and self.diag[k] > 0:
            k += 1
        self.k = k

        N = cp.Variable((self.n, self.n), symmetric = True)
        constraints = [N >= 0,
               self.Q - N >> 0.001] #for numerical help. might reduce applicability

        o_m1 = np.zeros((self.n, self.n))
        if self.maximize_sdp:
            for i in range(1, self.n + 1 - k):
                o_m1[-i, -i] = 1
        else:
            o_m1 = -np.ones((self.n, self.n))

        prob = cp.Problem(cp.Maximize(cp.trace(o_m1 @ N)), constraints)
        prob.solve(solver = cp.CVXOPT, verbose = True)
        if prob.status != "optimal":
            print(f"encountered difficulty while solving SDP ({prob.status})")
            if prob.status == "infeasible":
                #try again more aggressively numerically
                constraints = [N >= 0,
                               self.Q - N >> 0]
                prob = cp.Problem(cp.Maximize(cp.trace(o_m1 @ N)), constraints)
                prob.solve(solver = cp.CVXOPT, verbose = True)
                if prob.status != "optimal":
                    print(f"encountered difficulty while solving SDP again ({prob.status})")
                    sys.exit()
            else:
                sys.exit()

        print(f"time to solve SDP was {prob.solver_stats.solve_time}, with additional setup {prob.solver_stats.setup_time}. N =\n{N.value}")
        self.N = N.value
        self.S_psd = self.Q - self.N

        if self.maximize_sdp:
            #LDL without permutations. exists by psd property
            self.L_psd, self.diag_psd = ldl.LDL_without_pivot(self.S_psd)
            self.U_psd = self.L_psd.transpose()

        else:
            self.minima_for_bounds = [0.0] * self.n
            #convert type
            S_help = cp.psd_wrap(np.array(self.S_psd, dtype=float))
            for l in range(k, self.n):
                x = cp.Variable(self.n)
                prob = cp.Problem(cp.Minimize(cp.quad_form(x, S_help)), [x >= 0,
                                                                    x[l] == 1])
                prob.solve(solver = cp.OSQP, verbose = False)
                self.minima_for_bounds[l] = prob.value


    def update_difficult_bounds(self, j, lamb):
        if self.maximize_sdp:
            p_help_psd_part = 0
            p_help_N_part = 0

            for l in range(j + 1, self.n + 1):
                p_help_psd_part += self.U_psd[j - 1, l - 1] * self.coordinates[l - 1].current_pos
                p_help_N_part += self.N[j - 1, l - 1] * self.coordinates[l - 1].current_pos

            p = 1/(self.diag_psd[j - 1] + self.N[j - 1, j - 1]) * (self.diag_psd[j - 1] * p_help_psd_part + p_help_N_part)

            q_help_N_part = 0
            q_help_psd_part = 0

            for i in range(j + 1, self.n + 1):
                su_N = 0
                su_psd = self.coordinates[i - 1].current_pos
                for l in range(j + 1, self.n + 1):
                    su_N += self.N[i - 1, l - 1] * self.coordinates[l - 1].current_pos

                for l in range(i + 1, self.n + 1):
                    su_psd += self.U[i + 1, l + 1] * self.coordinates[l - 1].current_pos

                su_N *= self.coordinates[i - 1].current_pos
                q_help_N_part += su_N

                su_psd = self.diag_psd[i - 1] * su_psd ** 2
                q_help_psd_part += su_psd

            q = 1/(self.diag_psd[j - 1] + self.N[j - 1, j - 1]) * (q_help_N_part + q_help_psd_part - lamb)

            with np.errstate(invalid = 'raise'):
                try:
                    root_part = np.sqrt(p ** 2 - q)
                except:
                    if abs(p ** 2 - q) < 0.001:
                        root_part = 0
                    else:
                        print(p ** 2 - q)
                        print(f"at j = {j} numerical troubles. vector is {self.copy_of_current_coord()}")
                        print(f"p = {p}")
                        print(f"q = {q}")
                        print(f"might indicate that N_nn and diag_psd are very close to 0.")
                        sys.exit()

            lower_bound_exact = -p - root_part
            upper_bound_exact = -p + root_part

            lower_bound = np.ceil(lower_bound_exact).astype('int')
            upper_bound = np.floor(upper_bound_exact).astype('int')

            self.coordinates[j - 1].update_bounds(lower_bound, upper_bound)

        else:
            upper_bound_exact = np.sqrt(lamb / self.minima_for_bounds[j - 1])
            upper_bound = np.floor(upper_bound_exact).astype('int')

            if upper_bound_exact - upper_bound > 0.999: #rounding error??
                upper_bound += 1

            self.coordinates[j - 1].update_bounds(0, upper_bound)

    def update_easy_bounds(self, j, lamb):

        p_help = 0
        #k as a placeholder, not k from the difficult variables!
        for k in range(j + 1, self.n + 1):
            p_help += self.U[j - 1, k - 1] * self.coordinates[k - 1].current_pos

        q = p_help ** 2
        q_help = 0

        for i in range(j + 1, self.n + 1):
            su = self.coordinates[i - 1].current_pos
            for k in range(i + 1, self.n + 1):
                su += self.U[i - 1, k - 1] * self.coordinates[k - 1].current_pos
            q_help += self.diag[i - 1] * su ** 2

        q_help -= lamb
        q += 1/self.diag[j - 1] * q_help

        with np.errstate(invalid = 'raise'):
            try:
                root_part = np.sqrt(float((p_help) ** 2 - q))
                # print(root_part)
            except:
                if abs(p_help) ** 2 - q < 0.001:
                    root_part = 0
                else:
                    print(p_help ** 2 - q)
                    #this might happen since the bounds are not optimal anymore
                    print("easy bound problem")
                    sys.exit()


        lower_bound_exact = -p_help - root_part
        upper_bound_exact = -p_help + root_part

        lower_bound = np.ceil(lower_bound_exact)
        if not type(lower_bound) is int:
            lower_bound = lower_bound.astype('int')

        upper_bound = np.floor(upper_bound_exact)
        if not type(upper_bound) is int:
            upper_bound = np.floor(upper_bound_exact).astype('int')


        if upper_bound_exact - upper_bound > 0.999: #rounding error??
            upper_bound += 1

        if lower_bound - lower_bound_exact > 0.999: #needs better handling
            lower_bound -= 1


        self.coordinates[j - 1].update_bounds(lower_bound, upper_bound)

    #could be dropped if no special cone is used
    def get_cone_bounds_1(self, x):
        #cone inequalities lead to upper and lower bounds for y1
        #see notes for how these values come to be

        #TODO: See if this could introduce rounding errors

        lower, upper = -np.inf, np.inf

        for i in range(0, self.n):

            bound = -(self.C[i, 1:] @ np.matrix(x[1:]).transpose())[0,0]

            #sign of coefficient in front of y1 determines whether the bound is upper or lower

            if self.C[i, 0] < 0:
                upper = min(upper, bound / self.C[i, 0])

            elif self.C[i, 0] > 0:
                lower = max(lower, bound / self.C[i, 0])

            else: #coefficient is zero: ok if possible, otherwise rip
                if bound > 0:
                    print("x_1 impossible bound")
                    print(f"x: {x}")
                    return 0, -1 #impossible bounds

        return np.ceil(lower), np.floor(upper)

    #could be dropped, if no special cone is used
    def update_cone_bounds(self, j, lamb):
        if j == 1:
            lower_bound = self.coordinates[0].cone_lower_bound_trivial
            upper_bound = self.coordinates[0].cone_upper_bound_trivial

            #coordinate vector with first entry deleted
            x_end = np.delete(np.array(self.copy_of_current_coord()), 0, 0)
            #TODO: C_end and c do not change
            #remaining constraint matrix and coefficients before x_1
            C_end = self.C[:, 1:]
            c = self.C[:, 0]

            rhs = - C_end @ x_end

            #check every row and distinguish between the signs of the coefficient
            for i in range(0, self.n):
                if c[i] < 0:
                    upper_bound = min(upper_bound, rhs[i]/c[i])
                elif c[i] > 0:
                    lower_bound = max(lower_bound, rhs[i]/c[i])
                else: #coefficient 0
                    if rhs[i] > 0:
                        lower_bound = 0
                        upper_bound = -1
            #calculate bounds for x_1
            self.coordinates[0].update_cone_bounds(lower_bound, upper_bound)
        else:
            #do nothing - for now
            return

    def update_bounds(self, j, lamb):
        if j >= self.k + 1:
            self.update_difficult_bounds(j, lamb)
        else:
            self.update_easy_bounds(j, lamb)

    def go_next_coord(self, j):
        return self.coordinates[j - 1].go_next()

    def coord_is_in_bounds(self, j):
        return self.coordinates[j - 1].in_bounds()

    def match_coord_to_bound(self, j):
        if self.coordinates[j - 1].current_pos < self.coordinates[j - 1].lower_bound:
            self.coordinates[j - 1].reset_pos()

    def reset_coord(self, j):
        self.coordinates[j - 1].reset_pos()

    def zero_coordinates(self):
        if all(coord.current_pos == 0 for coord in self.coordinates):
            return True
        return False

    def coord_1(self):
        return self.coordinates[0].current_pos

    def evaluate(self):
        x_list = self.copy_of_current_coord()
        v = np.array(x_list)
        return v @ (self.Q @ v)

    def copy_of_current_coord(self):
        x_list = [coord.current_pos for coord in self.coordinates]
        return x_list

    def get_bounds_1(self):
        #this is a one dimensional quadratic eq., so there is necessarily only one range
        lower_bound = max(self.coordinates[0].cone_lower_bound, self.coordinates[0].lower_bound)
        upper_bound = min(self.coordinates[0].cone_upper_bound, self.coordinates[0].upper_bound)
        return lower_bound, upper_bound

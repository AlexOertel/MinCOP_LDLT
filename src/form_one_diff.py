import numpy as np
from fractions import Fraction
import sys
import ldl
import gurobipy as gp
from gurobipy import GRB

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
        self.Q_ = read_matrix(filename)
        self.n = self.Q_.shape[0]
        self.coordinates = []
        for i in range(0,self.n):
            self.coordinates.append(Coordinate())
        self.initialize_LLL()
        self.initialize_LDL()
        self.initialize_trivial_cone_bounds()
        # self.initialize_PSN_decomposition()
        self.initialize_difficult_bounds()

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
        #cone is just R^n_>=0

    #initialize the range of the variables based on the cone
    def initialize_trivial_cone_bounds(self):
        #R^n_>=0

        for j in range(0, self.n):
            self.coordinates[j].initialize_trivial_cone_bounds(0, np.inf)

    def initialize_difficult_bounds(self):
        #find how many bounds are difficult
        k = 0
        while k < self.n and self.diag[k] > 0:
            k += 1
        self.k = k

        self.minima_for_bounds = [0.0] * self.n

        #check if more than one difficult coordinate
        if k != self.n - 1:
            #only ok if psd:
            if any(self.diag[k:]):
                raise ValueError(f"too many difficult coordinates: {self.n-k} and not psd.")

        #calculate necessary minima for each difficult coordinate with convex quadratic programm
        for l in range(k, self.n):
            m = gp.Model()
            y = m.addMVar(shape = self.n) #y >= 0 continues by default

            m.addConstr(y[l] == 1)
            m.setObjective(y @ (self.Q @ y), GRB.MINIMIZE)

            m.optimize()

            self.minima_for_bounds[l] = m.ObjVal

            if m.ObjVal < -0.001:
                raise ValueError(f"In row {len(matrix_help)} wrong number of elements: {number_elements_in_row}, should be {n}")


    def update_difficult_bounds(self, j, lamb):
        #
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
            except:
                if abs(p_help ** 2 - q) < 0.001:
                    root_part = 0
                else:
                    #numerical trouble?
                    print(p_help ** 2 - q)
                    print(self.diag)
                    sys.exit()


        lower_bound_exact = -p_help - root_part
        upper_bound_exact = -p_help + root_part

        if upper_bound_exact < -0.001:
            self.coordinates[j - 1].update_bounds(1, 0)
        if lower_bound_exact < 0:
            lower_bound_exact = 0


        lower_bound = np.ceil(lower_bound_exact).astype('int')
        upper_bound = np.floor(upper_bound_exact).astype('int')

        if upper_bound_exact - upper_bound > 0.999: #rounding error??
            upper_bound += 1

        if lower_bound - lower_bound_exact > 0.999: #needs better handling
            lower_bound -= 1


        self.coordinates[j - 1].update_bounds(lower_bound, upper_bound)

    #unnecessary if no special cone is used
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

    def update_cone_bounds(self, j, lamb):
        if j == 1:
            lower_bound = self.coordinates[0].cone_lower_bound_trivial
            upper_bound = self.coordinates[0].cone_upper_bound_trivial

            #coordinate vector with first entry deleted
            x_end = np.delete(np.array(self.copy_of_current_coord()), 0, 0)
            #TODO: C_end und c aendern sich nie
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
            # print("cone bounds updated")
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

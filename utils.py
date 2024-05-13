"""
This file contains some utility functions for computing the performance bound
of LQ-MPC with model mismatch, the utility functions are used to compute several
baseline numerical quantities that are used to compute the final performance bound.

The naming convention follows that in the paper. For example, in the paper, we have
a function $g^{n}_{i,(x)}$ that computes the deviation of norms, then its naming is
[ fc_ec_gx ], meaning that it is a function, it is error-consistent, and it is with
function title g, and it is used to compute the deviation in state x. We list all the
functions that will be created here

From Lemma 5: deviation of matrix in state propagation
1. fc_ec_g_x --- $g^{(n)}_{i, (x)}$
2. fc_ec_g_u --- $g^{(n)}_{i, (u)}$

From Lemma 7: deviation of matrix from system level synthesis
3. fc_ec_bar_g_x --- $\bar{g}_{(x)}$
4. fc_ec_bar_g_u --- $\bar{g}_{(u)}$
5.1 fc_ec_theta_x --- $\theta_{N,(x)}$
5.2 fc_ec_theta_x_u --- $\theta_{N,(x,u)}$

From Lemma 1: exponential stability
6. ex_stability_lq --- it returns the value of $C^\ast_K$, $\lambda_K$, $\rho_K$, $\gamma$, and $\rho_\gamma$

From Proposition 1: three important terms
7. fc_ec_E_psi --- $E_{N, (\psi)}$
8. fc_ec_E_u --- $E_{N, (u)}$
9. fc_ec_E_psi_u --- $E_{N, (\psi,u)}$

From Proposition 2: energy-decreasing factors
10. geo_A --- $G_N(\hat{A})$
11. fc_omega_1 --- $\omega_{N, (1)}$
12. fc_omega_0d5 --- $\omega_{N, (0.5)}$
13. fc_ec_h --- $h$
"""
import matplotlib.pyplot as plt
import numpy as np
import random
# import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.linalg import cho_factor
import math

'''
Preliminaries
'''


def my_eigen(M):
    """
    :param M: The M matrix
    :return: A dictionary that contains the max and min eigenvalues of the matrix, and the condition number
    """

    # Compute the eigenvalues of the matrix M
    eigenvalues = np.linalg.eigvals(M)

    # Find the maximum and minimum eigenvalues
    max_eigenvalue = np.max(eigenvalues)
    min_eigenvalue = np.min(eigenvalues)

    # compute the condition number
    ratio_M = max_eigenvalue / min_eigenvalue

    return {'max': max_eigenvalue, 'min': min_eigenvalue, 'ratio': ratio_M}


'''
From Lemma 5: deviation of matrix in state propagation
1. fc_ec_g_x --- $g^{(n)}_{i, (x)}$
2. fc_ec_g_u --- $g^{(n)}_{i, (u)}$
'''


def fc_ec_g_x(n, i, e_A, f_A):
    """
    Compute the error-consistent function g^{(n)}_{i, (x)}
    :param n: the power
    :param i: the index of the open-loop
    :param e_A: matrix normed error ||A - \hat{A}||_F = eA
    :param f_A: matrix norm of A ||\hat{A}|| = fA
    :return: the value of the error-consistent function g^{(n)}_{i, (x)}
    """

    # compute the perturbed norm
    sum_norm = e_A + f_A

    # compute the difference without the final power n
    mybase = sum_norm ** i - f_A ** i

    # return the final value
    return mybase ** n


def fc_ec_g_u(n, i, e_A, f_A, e_B, f_B):
    """
    Compute the error-consistent function g^{(n)}_{i, (u)}
    :param n: the power
    :param i: the index of the open-loop
    :param e_A: matrix normed error ||A - \hat{A}||_F = e_A
    :param f_A: matrix norm of A ||\hat{A}|| = f_A
    :param e_B: matrix normed error ||B - \hat{B}||_F = e_B
    :param f_B: matrix norm of B ||\hat{B}|| = f_B
    :return: the value of the error-consistent function g^{(n)}_{i, (u)}
    """

    # compute the error due to difference in A
    err_1 = (e_B + f_B) * fc_ec_g_x(1, i, e_A, f_A)

    # compute the error due to difference in B
    err_2 = e_B * (f_A ** i)

    # return the final value
    return (err_1 + err_2) ** n


'''
Auxiliary functions that create two big matrices for system_level synthesis of linear systems
bar_x = Phi * x + Gamma * bar_u, where x is the initial state
'''


def sl_syn_Phi(N, A):
    """
    This function generates the Phi matrix in the system-level synthesis
    based on x^+ = A * x + B * u
    :param A: A
    :param N: prediction horizon
    :return: Phi: [I; A; A^2; ...; A^N]
    """
    # Initialize a list
    matrices_list = [np.eye(A.shape[0])]

    # computation of A^k for k = 1 to N
    for k in range(1, N + 1):
        matrices_list.append(np.linalg.matrix_power(A, k))

    sl_Phi = np.vstack(matrices_list)
    return sl_Phi


def sl_syn_Gamma(N, A, B):
    """
    This function generates the Gamma matrix in the system-level synthesis
    :param N: Prediction horizon
    :param A: system matrix A
    :param B: system matrix B
    :return: Gamma: [[0,0,...,0], [B, 0, ..., 0], [AB, B, 0, ..., 0], ..., [A^{N-1}B, A^{N-2}B, ..., B]]
    """
    # Get the dimensions of matrices A and B
    m, A_col = A.shape
    B_row, n = B.shape

    # Initialize the big matrix with zeros
    big_matrix = np.zeros((N * m, N * n))

    # Set B in the appropriate positions
    for i in range(N):
        big_matrix[i * m:(i + 1) * m, i * n:(i + 1) * n] = B

    # Compute and set the values of the rest of the matrix
    for i in range(1, N):
        power_A = np.linalg.matrix_power(A, i)
        for j in range(i, N):
            big_matrix[j * m:(j + 1) * m, (j - i) * n:(j - i + 1) * n] = power_A @ B

    first_row = np.zeros((m, N * n))

    sl_Gamma = np.vstack([first_row, big_matrix])

    return sl_Gamma


'''
From Lemma 7: deviation of matrix from system level synthesis
3. fc_ec_bar_g_x --- $bar{g}_{(x)}$
4. fc_ec_bar_g_u --- $bar{g}_{(u)}$
5.1 fc_ec_theta_x --- $theta_{N,(x)}$
5.2 fc_ec_theta_xu --- $theta_{N,(x,u)}$
'''


def fc_ec_bar_g_x(N, e_A, f_A):
    """
    The function computes the \bar{g}_{(x)} function,
    :param N: Prediction horizon
    :param e_A: mismatch in matrix A
    :param f_A: norm of matrix A
    :return: value of \bar{g}_{(x)} using the value of g_{i, (x)}
    """
    # initialize the summation
    my_out = 0

    # compute the summation by adding the term g
    for i in range(N):
        my_out += fc_ec_g_x(1, i + 1, e_A, f_A)

    return my_out


def fc_ec_bar_g_u(N, e_A, f_A, e_B, f_B):
    """
    The function computes the \bar{g}_{(u)} function
    :param N: Prediction horizon
    :param e_A: mismatch in matrix A
    :param f_A: norm of matrix A
    :param e_B: mismatch in matrix B
    :param f_B: norm of matrix B
    :return: value of \bar{g}_{(u)} using the value of g_{i, (u)}
    """
    # initialize the summation
    sum_out = 0
    sum_in = 0
    for i in range(N):
        # update the inner summation
        sum_in += fc_ec_g_u(1, i, e_A, f_A, e_B, f_B)
        # update the outer summation
        sum_out += sum_in

    return sum_out


def fc_ec_theta(N, e_A, e_B, A, B, maxQ):
    """
    The function computes the theta_{(x)}, and theta_{(x,u)} functions
    :param N: prediction horizon
    :param e_A: mismatch in matrix A
    :param e_B: mismatch in matrix B
    :param A: matrix A
    :param B: matrix B
    :param maxQ: the maximum eigenvalue of matrix Q
    :return: a dictionary with the values of theta_{(x)} and theta_{(x, u)}
    """
    # compute the norm of matrices A and B
    f_A = np.linalg.norm(A, ord=2)
    f_B = np.linalg.norm(B, ord=2)

    # form the matrix Gamma
    my_gamma = sl_syn_Gamma(N, A, B)
    # form the matrix Phi
    my_phi = sl_syn_Phi(N, A)

    # compute my bar_x
    my_bar_x = fc_ec_bar_g_x(N, e_A, f_A)

    # compute my bar_u
    my_bar_u = fc_ec_bar_g_u(N, e_A, f_A, e_B, f_B)

    # computation of theta_x
    err_1 = np.linalg.norm(my_gamma, ord=2) * my_bar_u
    err_2 = my_bar_u ** 2
    my_theta_u = maxQ * (2 * err_1 + err_2)

    # computation of theta_u
    err_3 = np.linalg.norm(my_gamma, ord=2) * my_bar_x
    err_4 = (np.linalg.norm(my_phi, ord=2)
             * my_bar_u)
    err_5 = my_bar_x * my_bar_u
    my_theta_x_u = maxQ * (err_3 + err_4 + err_5)

    return {'theta_u': my_theta_u, 'theta_x_u': my_theta_x_u}


def fc_ec_E(N, e_A, e_B, A, B, Q, R, x, bar_u, bar_d_u):
    """
    This function computes the error-consistent terms E_{N,(\psi)}, E_{N,(u)}, and E_{N,(\psi, u)}
    :param N: prediction horizon
    :param e_A: mismatch in matrix A
    :param e_B: mismatch in matrix B
    :param A: matrix A
    :param B: matrix B
    :param Q: matrix Q
    :param R: matrix R
    :param x: the initial state
    :param bar_u: the solution to a simple QP that returns the input two norm
    :param bar_d_u: the solution to a simple QP that returns the difference input norm
    :return: a dictionary with the value of the error-consistent terms E_{N,(\psi)}, E_{N,(u)}, and E_{N,(\psi, u)}
    """

    # obtaining the information of matrix Q and R
    info_Q = my_eigen(Q)
    info_R = my_eigen(R)

    # obtaining the matrix norm A and B
    f_A = np.linalg.norm(A, ord=2)
    f_B = np.linalg.norm(B, ord=2)

    # obtaining the norm of x
    norm_x = np.linalg.norm(x, ord=2)

    # -------------- computing the term E_{N,(\psi)} --------------

    sum_in = 0  # initialize the inner sum
    sum_out = 0  # initialize the outer sum
    for i in range(N + 1):
        # first update the outer sum
        sum_out += (sum_in + fc_ec_g_x(2, i, e_A, f_A)) * ((norm_x ** 2) + i * bar_u)
        # then update the inner sum
        sum_in += fc_ec_g_u(2, i, e_A, f_A, e_B, f_B)

    # final computation
    my_E_psi = info_Q['max'] * sum_out

    # -------------- computing the term E_{N, (u)} ----------------

    # get the information of theta functions
    info_theta = fc_ec_theta(N, e_A, e_B, A, B, info_Q['max'])

    # compute the term bar_theta
    my_bar_theta = math.sqrt(N * bar_u) * info_theta['theta_u'] + np.linalg.norm(x, ord=2) * info_theta['theta_x_u']

    # ----- formulate the big matrix hatH_N -----
    my_gamma = sl_syn_Gamma(N, A, B)  # form gamma
    barQ = np.kron(Q, np.eye(N + 1))  # form barQ_{N+1}
    barR = np.kron(R, np.eye(N))  # form barR_{N}
    hatH = barR + my_gamma.T @ barQ @ my_gamma  # compute the final hatH

    # compute the minimum eigenvalue of hatH
    min_H = np.min(np.linalg.eigvals(hatH))

    # final computation
    my_E_u = info_R['max'] * (np.min([math.sqrt(N * bar_d_u), my_bar_theta / min_H]) ** 2)

    # ----------- computing the term E_{N, (\psi, u)} -------------
    my_bar_g_u = fc_ec_bar_g_u(N, e_A, f_A, e_B, f_B)  # form bar_g_u
    norm_gamma = np.linalg.norm(my_gamma, ord=2)  # compute the norm of the desired matrix

    my_E_psi_u = info_Q['max'] / info_R['max'] * ((norm_gamma + my_bar_g_u) ** 2) * my_E_u

    # return the results
    return {'E_psi': my_E_psi, 'E_u': my_E_u, 'E_psi_u': my_E_psi_u}


'''
From Lemma 1: exponential stability
6. ex_stability_lq --- it returns the value of $C^ast_K$, $\lambda_K$, $rho_K$, $\gamma$, and $rho_\gamma$
'''


def ex_stability_lq(A, B, Q, R, K):
    """
    This function computes several quantities in the exponential stability results
    :param A: matrix A
    :param B: matrix B
    :param Q: matrix Q
    :param R: matrix R
    :param K: matrix K, the linear control law
    :return: a dictionary with the values of $C^ast_K$, $\lambda_K$, $rho_K$, $\gamma$, and $rho_\gamma$
    """
    # get the norm of matrix K
    normK = np.linalg.norm(K, ord=2)
    # compute the closed-loop matrix
    A_cl = A + B * K
    # computing \rho_K
    rho_K = (np.max(np.abs(np.linalg.eigvals(A_cl))) + 0.4) ** 2

    # computing \lambda_K
    # myH = eigenvectors
    # invH = np.linalg.inv(myH)
    # obtained from basic_test_2
    lambda_K_2 = 1.21

    # obtaining the eigen information of matrices Q and R
    infoQ = my_eigen(Q)
    infoR = my_eigen(R)

    # computing C^\ast_K
    my_scale = 1 + infoR['max'] * (normK ** 2) / (infoQ['min'])
    my_C_star = my_scale * np.max([1, infoQ['ratio'] * lambda_K_2])

    # computing \gamma
    my_gamma = my_C_star / (1 - rho_K)

    # computing \rho_\gamma
    my_rho_gamma = (my_gamma - 1) / my_gamma

    return {'C_K': my_C_star, 'lambda_K': lambda_K_2, 'rho_K': rho_K, 'gamma': my_gamma, 'rho_gamma': my_rho_gamma}


'''
From Proposition 2: energy-decreasing factors
10. geo_M --- computation of geometric sum of a matrix norm
11.1. fc_omega_1 --- $\omega_{N, (1)}$
11.2. fc_omega_0d5 --- $\omega_{N, (0.5)}$
11.3. fc_eta --- $\eta_N$
12. fc_ec_h --- $h$
'''


def geo_M(M, n):
    """
    :param M: The M matrix
    :param n: The power (the maximum power is 2*(n-1))
    :return: the geometric sum of the norm of the matrix: \|M\|^0, \|M\|^(1*2), ..., \|M\|^{2*(n-1)}
    """

    # get the two norm
    norm_A_two = np.linalg.norm(M, ord=2)

    # compute the geometric series based on whether the norm is 1 or not
    if norm_A_two == 1:
        output = n
    else:
        output = (1 - norm_A_two ** (2 * n)) / (1 - norm_A_two ** 2)

    return output


def fc_omega_eta_extension(N, A, B, Q, R, K, hatK, L_V, N_0):
    """
    This function computes the omega function terms used in computing the performance bound in proposition 2
    :param N: prediction horizon
    :param A: matrix A
    :param B: matrix B
    :param Q: matrix Q
    :param R: matrix R
    :param K: linear control gain for the estimated open-loop control
    :param hatK: linear control gain for the deviated estimated open-loop control
    :param L_V: bounding coefficient for the open-loop MPC value function
    :param N_0: minimum prediction horizon
    :return: a dictionary with the values of the two omega functions
    """
    # ----------------Preparation-----------------
    # computing the norm of matrix A
    normA = np.linalg.norm(A, ord=2)

    # obtaining the closed-loop matrix
    A_cl = A + B * K
    # computing the norm of the closed loop matrix
    normA_cl = np.linalg.norm(A_cl, ord=2)

    # getting the information of matrix Q
    info_Q = my_eigen(Q)

    # -------------- Computation details ----------------
    # computing the geometric sum
    G_A = geo_M(A, N - 1)

    # computing the relevant quantities of the exponential stability
    info_Stable = ex_stability_lq(A, B, Q, R, K)
    info_Stable_deviate = ex_stability_lq(A, B, Q, R, hatK)

    # computing an intermediate quantity stemmed from terminal cost propagation
    my_term = info_Stable_deviate['C_K'] + (normA_cl ** 2) * info_Q['ratio']

    # computing the minimum prediction horizon
    N_min = N_0 - math.log((my_term - 1) * info_Stable['gamma']) / math.log(info_Stable['rho_gamma'])

    # computation of omega_{N, (1)}
    my_omega_N1 = info_Q['max'] * (my_term * (normA ** (2 * N - 2)) + G_A)

    # computation of omega_{N, (0.5)}
    my_decay = info_Q['max'] * (normA ** (2 * N - 2)) * info_Stable['gamma'] * (info_Stable['rho_gamma'] ** (N - N_0))
    my_omega_N0d5 = math.sqrt(info_Q['max'] * (L_V - 1) * G_A) + 0.5 * my_term * math.sqrt(my_decay)

    # computation of eta
    my_eta = (my_term - 1) * info_Stable['gamma'] * (info_Stable['rho_gamma'] ** (N - N_0))

    # computing the error_threshold
    my_err_threshold = ((math.sqrt(my_omega_N0d5 ** 2 + my_omega_N1 * (1 - my_eta)) - my_omega_N0d5) / my_omega_N1) ** 2

    return {'omega_N1': my_omega_N1, 'omega_N0d5': my_omega_N0d5,
            'eta': my_eta, 'err_th': my_err_threshold, 'N_min': N_min}


def fc_omega_eta(N, A, B, Q, R, K, L_V, N_0):
    """
    This function computes the omega function terms used in computing the performance bound in proposition 2
    :param N: prediction horizon
    :param A: matrix A
    :param B: matrix B
    :param Q: matrix Q
    :param R: matrix R
    :param K: linear control gain for the estimated open-loop control
    :param L_V: bounding coefficient for the open-loop MPC value function
    :param N_0: minimum prediction horizon
    :return: a dictionary with the values of the two omega functions
    """
    # ----------------Preparation-----------------
    # computing the norm of matrix A
    normA = np.linalg.norm(A, ord=2)

    # obtaining the closed-loop matrix
    # A_cl = A + B * K
    # computing the norm of the closed loop matrix
    # normA_cl = np.linalg.norm(A_cl, ord=2)

    # getting the information of matrix Q
    info_Q = my_eigen(Q)

    # -------------- Computation details ----------------
    # computing the geometric sum
    G_A = geo_M(A, N - 1)

    # computing the relevant quantities of the exponential stability
    info_Stable = ex_stability_lq(A, B, Q, R, K)
    # info_Stable_deviate = ex_stability_lq(A, B, Q, R, hatK)

    # computing an intermediate quantity stemmed from terminal cost propagation
    my_term = 1 + (normA ** 2) * info_Q['ratio']

    # computing the minimum prediction horizon
    N_min = math.ceil((N_0 - math.log((normA ** 2) * info_Q['ratio'] * info_Stable['gamma'])
                       / math.log(info_Stable['rho_gamma'])))

    # computation of omega_{N, (1)}
    my_omega_N1 = info_Q['max'] * (my_term * (normA ** (2 * N - 2)) + G_A)

    # computation of omega_{N, (0.5)}
    my_decay = info_Q['max'] * (normA ** (2 * N - 2)) * info_Stable['gamma'] * (info_Stable['rho_gamma'] ** (N - N_0))
    my_omega_N0d5 = math.sqrt(info_Q['max'] * (L_V - 1) * G_A) + 0.5 * my_term * math.sqrt(my_decay)

    # computation of eta
    my_eta = (my_term - 1) * info_Stable['gamma'] * (info_Stable['rho_gamma'] ** (N - N_0))

    # computing the error_threshold
    my_err_threshold = ((math.sqrt(my_omega_N0d5 ** 2 + my_omega_N1 * (1 - my_eta)) - my_omega_N0d5) / my_omega_N1) ** 2

    return {'omega_N1': my_omega_N1, 'omega_N0d5': my_omega_N0d5,
            'eta': my_eta, 'err_th': my_err_threshold, 'N_min': N_min}


def fc_ec_h(e_A, e_B, Q, R):
    """
    This function calculates the error-consistent function in the stability analysis
    :param e_A: mismatch in A
    :param e_B: mismatch in B
    :param Q: matrix Q
    :param R: matrix R
    :return: the value of the error-consistent function h
    """
    info_Q = my_eigen(Q)
    info_R = my_eigen(R)

    return (e_A ** 2) / info_Q['min'] + (e_B ** 2) / info_R['min']


'''
We still needs some additional functions to help proceeding Lemma 2
13. A function that computes \epsilon_K given K and the input constraint set
14. The bounding functions that finds the scalars L_V and N_0
'''


def local_radius(F_u, K, Q):
    """
    This function computes the value of \epsilon_K given K and the input constraint set F_u * u \leq vex{1}
    :param F_u: The input constraint matrix
    :param K: The linear feedback gain
    :param Q: matrix Q
    :return: the value of \epsilon_K
    """

    # initialize a vector to store the desired norm in the denominator as \|F_u*K_{i,:}\|^2_{inv{Q}}
    a = np.zeros(F_u.shape[0], dtype=float)
    M = F_u @ K
    invQ = np.linalg.inv(Q)
    for i in range(M.shape[0]):
        a[i] = M[i:i + 1] @ invQ @ M[i:i + 1].T

    return 1 / np.max(a)


def ex_stability_bounds(gamma, epsilon_K, M_V):
    """
    This function computes the scalars L_V and N_0 in the local exponential stability property
    :param gamma: the coefficient returned by ex_stability_lq
    :param epsilon_K: the radius of the local ellipsoid
    :param M_V: a given upper bound of the MPC value function
    :return: a dictionary with two values for L_V, N_0, and N_min
    """
    myL_V = np.max([gamma, M_V / epsilon_K])
    myN_0 = math.ceil(np.max([0, M_V / epsilon_K - gamma]))

    # info_Q = my_eigen(Q)

    # my_num = math.log((np.linalg.norm(A, ord=2) ** 2) * info_Q['ratio'] * gamma)
    # my_den = math.log(rho_gamma)
    # myN_min = myN_0 + 1 - my_num / my_den

    return {'L_V': myL_V, 'N_0': myN_0}


'''
We need a final function that computes the value of bar_u, and bar_d_u, which requires solving optimization problems
'''


def bar_u_solve(F_u):
    """
    This function computes the value of the maximum u
    :param F_u: The matrix defining the polytopic constraints
    :return: The optimal solution
    """
    n_constr = F_u.shape[0]
    dim_u = F_u.shape[1]

    # create the model
    model = gp.Model("maximum-norm")
    # disable the printing information
    model.setParam('OutputFlag', 0)

    # define the decision variable
    u = model.addVars(dim_u, lb=-GRB.INFINITY, name="u")

    # Set objective function: maximize the squared Euclidean norm
    model.setObjective(gp.quicksum(u[i] * u[i] for i in range(dim_u)), GRB.MAXIMIZE)

    # Add constraint F_u * u <= 1
    for i in range(n_constr):
        model.addConstr(gp.quicksum(F_u[i, j] * u[j] for j in range(dim_u)) <= 1, name="u_constr" + str(i))

    # Solve the problem
    model.optimize()

    return model.objVal


def bar_d_u_solve(F_u):
    """
    This function computes the value of the maximum difference
    :param F_u: The matrix defining the polytopic constraints
    :return: The optimal solution
    """
    n_constr = F_u.shape[0]
    dim_u = F_u.shape[1]

    # Define the model
    model = gp.Model("maximum-difference")
    # disable the printing information
    model.setParam('OutputFlag', 0)

    # Create variables
    u_1 = model.addVars(dim_u, lb=-GRB.INFINITY, name="u_1")
    u_2 = model.addVars(dim_u, lb=-GRB.INFINITY, name="u_2")

    # Set objective
    model.setObjective(gp.quicksum((u_1[i] - u_2[i]) * (u_1[i] - u_2[i]) for i in range(dim_u)), GRB.MAXIMIZE)

    for i in range(n_constr):
        model.addConstr(gp.quicksum(F_u[i, j] * u_1[j] for j in range(dim_u)) <= 1, name="u_constr" + str(i))
        model.addConstr(gp.quicksum(F_u[i, j] * u_2[j] for j in range(dim_u)) <= 1, name="u_constr" + str(i + n_constr))

    # Optimize
    model.optimize()

    return model.objVal


"""
Some additional functions that support the miscellaneous requirements in the main simulation
"""


def rot_2D(theta):
    """
    This function creates a rotation matrix in 2D
    :param theta: the rotation angle
    :return: a rotation matrix in 2D as a 2D numpy array
    """
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    return R


def rot_action_2D(x, theta):
    """
    This function creates the rotated 2D vector based on all the values in theta
    :param x: the vector that is to be rotated
    :param theta: a 1D numpy array that contains all the values in theta
    :return: 2D numpy array that contains all the rotated vectors
    """
    my_out = np.zeros([2, theta.shape[0]])
    for i in range(theta.shape[0]):
        my_out[:, i:i + 1] = rot_2D(theta[i]) @ x

    return my_out


def circle_generator(N_points, ratio_ext_radius, my_base, Q):
    """
    This function generates a set of states on a circle
    :param N_points: The number of data points on a circle
    :param ratio_ext_radius: The extending ratio of the radius
    :param my_base: x^T * Q * x <= my_base, which defines the size of the local region
    :param Q: The matrix Q of the stage cost
    :return: A 2-by-N_points numpy array
    """
    # Cholesky decomposition of the matrix Q
    root_Q, _ = cho_factor(Q)

    # Compute the base state (a 2D vector)
    x0_base = np.array([[ratio_ext_radius * math.sqrt(my_base)], [0.0]])

    # Specify the rotation angles for rotating the initial state
    my_theta = np.linspace(0, 2 * (1 - 1 / N_points) * math.pi, N_points)

    # Computing the set of initial vectors that will be used
    x0_vec = np.linalg.inv(root_Q) @ rot_action_2D(x0_base, my_theta)

    return x0_vec


"""
For testing purposes, I need some additional functions that are useful for simplifying the test
"""


def N_incremental_test(A, Q, gamma, rho_gamma):
    info_Q = my_eigen(Q)
    my_num = math.log((np.linalg.norm(A, ord=2) ** 2) * info_Q['ratio'] * gamma)
    my_den = math.log(rho_gamma)
    myN_min = math.ceil(- my_num / my_den)

    return myN_min


"""
Some additional functions for plotting
"""


def default_color_generator():
    """
    This function returns the default colors used in matplotlib
    :return:
    """
    my_color_dict = {'C0': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                     'C1': (1.0, 0.4980392156862745, 0.054901960784313725),
                     'C2': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                     'C3': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                     'C4': (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                     'C5': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                     'C6': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                     'C7': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                     'C8': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                     'C9': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)}

    return my_color_dict


def gradient_color(Value, color_base):

    min_Value = Value.min()
    max_Value = Value.max()
    colors_z = ['color' for _ in range(len(Value.flatten()))]
    for i, z_val in enumerate(Value.flatten()):
        normalized_z = (z_val - min_Value) / (max_Value - min_Value)
        colors_z[i] = (normalized_z * color_base[0],
                       normalized_z * color_base[1],
                       normalized_z * color_base[2],
                       1)  # (R, G, B, alpha)

    return colors_z


def generate_random_matrix(rows, cols, a, b):
    """
    This is a function that generates a random matrix whose entry obeys the uniform distribution
    from [a, b]
    :param rows: number of rows
    :param cols: number of cols
    :param a: the left bound
    :param b: the right bound
    :return: A random matrix
    """
    matrix = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(random.uniform(a, b))
        matrix.append(row)
    return np.array(matrix)


def random_matrix(M: np.ndarray, N_matrix: int,
                  norm_bound, norm_type: str) -> np.ndarray:
    """
    This function creates a bunch of random matrix satisfying a specific norm
    :param M: The baseline matrix
    :param N_matrix: The number of required matrix
    :param norm_type: character, either 'f', '2'
    :param norm_bound: the error bound
    :return: A bunch of matrix 3-dimensional, M.shape[0] -by- M.shape[1] -by- N_matrix
    """
    # Get the row and column of the desired matrix
    my_row = M.shape[0]
    my_col = M.shape[1]

    # Set the counter to be 0
    set_counter = 0

    # Initialize the output matrix
    out_set_matrices = np.zeros([my_row, my_col, 5 * N_matrix])

    # Set the loop
    if norm_type == 'f':
        while set_counter < N_matrix:
            temp_M = generate_random_matrix(my_row, my_col, -norm_bound, norm_bound)
            if np.isclose(np.linalg.norm(temp_M), norm_bound):
                out_set_matrices[:, :, set_counter] = temp_M
                set_counter += 1
        while set_counter < 5 * N_matrix:
            temp_M = generate_random_matrix(my_row, my_col, -norm_bound, norm_bound)
            if np.linalg.norm(temp_M) <= norm_bound:
                out_set_matrices[:, :, set_counter] = temp_M
                set_counter += 1
    elif norm_type == '2':
        while set_counter < N_matrix:
            temp_M = generate_random_matrix(my_row, my_col, -norm_bound, norm_bound)
            if np.isclose(np.linalg.norm(temp_M, ord=2), norm_bound):
                out_set_matrices[:, :, set_counter] = temp_M
                set_counter += 1
        while set_counter < 5 * N_matrix:
            temp_M = generate_random_matrix(my_row, my_col, -norm_bound, norm_bound)
            if np.linalg.norm(temp_M, ord=2) <= norm_bound:
                out_set_matrices[:, :, set_counter] = temp_M
                set_counter += 1

    return out_set_matrices


def error_matrix_generator(A: np.ndarray, B: np.ndarray,
                           error_vec: np.ndarray, N_matrix: int, norm_type) -> dict:
    """
    This function generates error matrix for different error level
    :param A: The matrix A
    :param B: The matrix B
    :param error_vec: the error vector specifying all the required error levels
    :param N_matrix: The number of matrix for each error level
    :param norm_type: The type of considered norm
    :return: A dictionary that contains two elements, A and B
             each of which is a 4D numpy array
    """
    output_A = np.zeros([A.shape[0], A.shape[1], 5 * N_matrix, len(error_vec)])
    output_B = np.zeros([B.shape[0], B.shape[1], 5 * N_matrix, len(error_vec)])
    for i in range(len(error_vec)):
        output_A[:, :, :, i] = random_matrix(A, N_matrix, error_vec[i], norm_type)
        output_B[:, :, :, i] = random_matrix(B, N_matrix, error_vec[i], norm_type)

    np.save('error_A' + '_' + norm_type + '.npy', output_A)
    np.save('error_B' + '_' + norm_type + '.npy', output_B)

    return {'error_A': output_A, 'error_B': output_B}


'''
I need some functions for plotting data, it will simplify the main plotting function
'''


def statistical_continuous(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray,
                           info_text: dict, info_color: tuple,
                           font_type: str, font_size: dict,
                           x_scale_log=False, y_scale_log=False, set_x_ticks=False) -> None:
    """
    This function plots a statistical data series
    :param ax: the handle of the subplots, e.g., ax[1, 0] (the second row, first column)
    :param x_data: the x-axis data (1-D array)
    :param y_data: a bunch of y-axis data (2-D array)
    :param info_text: the text information, a dictionary that contains labels and titles
    :param info_color: color information, just a color
    :param font_type: the type of the used font
    :param font_size: the size of the used font, a dictionary that contains the font size
                      for label, title, and legend
    :param x_scale_log: whether the x-axis is in log
    :param y_scale_log: whether the y-axis is in log
    :param set_x_ticks: whether to specify the x_ticks
    :return: NONE (simply execute a set of commands)
    """
    # Compute the max, min, mena and variance
    y_max = np.max(y_data, axis=0)
    y_min = np.min(y_data, axis=0)
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)

    # Creat color variations
    color_bound = tuple(x * 0.75 for x in info_color)
    color_variance = tuple(x * 0.5 for x in info_color)
    color_range = tuple(x * 0.25 for x in info_color)

    # basic mean plot
    ax.plot(x_data, y_mean,
            label=info_text['data'],
            linewidth=2.5, color=info_color)
    # plot the max and min
    ax.plot(x_data, y_min,
            linewidth=1.5, linestyle=':', color=color_bound)
    ax.plot(x_data, y_max,
            linewidth=1.5, linestyle=':', color=color_bound)
    # plot the max and min
    ax.plot(x_data, y_mean - y_std,
            linewidth=1.5, linestyle='--', color=color_bound)
    ax.plot(x_data, y_mean + y_std,
            linewidth=1.5, linestyle='--', color=color_bound)

    # plot the variance (fill the shaded color)
    ax.fill_between(x_data, y_mean - y_std, y_mean + y_std, color=color_variance, alpha=0.25)

    # plot the max and the min (fill the shaded color)
    ax.fill_between(x_data, y_min, y_max, color=color_range, alpha=0.125)

    # set the title and the labels
    ax.set_title(info_text['title'],
                 fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
    ax.set_xlabel(info_text['x_label'],
                  fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
    # set the x-axis
    if set_x_ticks:
        ax.set_xticks(x_data)
    # ax.set_ylabel('Y Label')

    # set the legend
    ax.legend(loc='upper left', fontsize=font_size["legend"],
              prop={'family': font_type, 'size': font_size["legend"]})

    # set the log-scale
    if x_scale_log:
        ax.set_xscale('log')
    if y_scale_log:
        ax.set_yscale('log')

    # set the background and grid
    ax.set_facecolor((0.95, 0.95, 0.95))
    ax.grid(True, linestyle='--', color='white', linewidth=1)

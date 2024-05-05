"""
This is the script for testing the LQ_MPC_Calculator and also for determining a good prediction horizon
as well as the maximum level of permitted error
"""

import numpy as np
import control as ct
from utils import local_radius, rot_action_2D, ex_stability_lq, ex_stability_bounds, fc_omega_eta, circle_generator
from scipy.linalg import cho_factor
# import numpy.linalg as la
import math
from utils_class import LQ_MPC_Controller, LQ_RDP_Calculator, LQ_RDP_Behavior

"""
This is the file for determining several important parameters for simulation
"""
# --------------Initialization---------------

# Specify the estimated system -- We use marginal stable
A = np.array([[1, 0.7], [0.12, 0.4]])
B = np.array([[1], [1.2]])

# Specify the cost -- We use the identity as the most simple case
Q = 2 * np.eye(2)
R = np.eye(1)

# Specify the input constraints -- We simply confine -1 <= u <= 1
F_u = np.array([[10], [-10]])

# Specify the number of the initial states that will be simulated
N_init = 8

# Specify the rotation angles for rotating the initial state
# my_theta = np.linspace(0, math.pi, N_init)

# ---------------Prerequisite-----------------

# Computing the LQR optimal gain -- Remark: the convention is A - BK
K_lqr, P_lqr, eig_lqr = ct.dlqr(A, B, Q, R)

# Computing the terminal set for the LQR gain
epsilon_lqr = local_radius(F_u, -K_lqr, Q)

# Specifying the extension factor -- Remark: this factor must be greater than 1
ratio_x0 = 1.5

# Computing the set of initial vectors that will be used
x0_vec = circle_generator(N_init, ratio_x0, epsilon_lqr, Q)

print(x0_vec)

# ------------- Computing the energy bar M_{\hat{V}} -------------
# A test horizon
N_horizon_test = 6

# Defining reference trajectory
x_ref = np.zeros((2, N_horizon_test))  # Example reference trajectory
u_ref = np.zeros((1, N_horizon_test))  # Example reference control inputs

# Create the test MPC controller
# here P is set as Q since we do not have a terminal cost
MPC_test = LQ_MPC_Controller(N_horizon_test, A, B, Q, R, Q, F_u)

my_behavior = LQ_RDP_Behavior(A, B, Q, R, F_u, -K_lqr, 6, 10, -6, -2)

M_V_test = my_behavior.OL_energy_bound(N_horizon_test, 8, ratio_x0, x_ref, u_ref)

info_lqr_ex = ex_stability_lq(A, B, Q, R, -K_lqr)

info_lqr_bar = ex_stability_bounds(info_lqr_ex['gamma'], epsilon_lqr, M_V_test)

info_error = fc_omega_eta(N_horizon_test, A, B, Q, R, -K_lqr, info_lqr_bar['L_V'], info_lqr_bar['N_0'])

"""
Print the tested result to see whether it works
"""

print(f"The chosen energy bound is", M_V_test)
print(f"The spectral radius of matrix A is", {np.max(np.linalg.eigvals(A))})
print("\n")
print(f"The coefficient C_K", {info_lqr_ex['C_K']})
print(f"The coefficient lambda_K", {info_lqr_ex['lambda_K']})
print(f"The coefficient rho_K", {info_lqr_ex['rho_K']})
print(f"The coefficient gamma", {info_lqr_ex['gamma']})
print(f"The coefficient rho_gamma", {info_lqr_ex['rho_gamma']})
print("\n")
print(f"The radius epsilon corresponding to the LQR is", {epsilon_lqr})
print(f"The critical prediction horizon is", {info_lqr_bar['N_0']})
print(f"The minimum required prediction horizon:", info_error['N_min'])
print('The used horizon:', N_horizon_test)
print('The error threshold:', info_error['err_th'])

"""
Print the result to see whether the energy is indeed decreasing
"""

# choose the error bound to be smaller than the threshold
e_A = 0.01
e_B = 0.01
my_calculator = LQ_RDP_Calculator(A, B, Q, R, F_u)
info_decrease = my_calculator.energy_decreasing(N_horizon_test, e_A, e_B, -K_lqr, M_V_test)

# choose the pairs (p_i, q_i)
p = np.array([0.1, 1, 0.6])  # a trivial default choice
info_bound = my_calculator.energy_bound(N_horizon_test, e_A, e_B, 1 * x0_vec[:, 0], p)

print(info_decrease)
print(info_bound)

# info_xi = my_behavior.data_generation_xi(-K_lqr, M_V_test, N_horizon_test, 0.01)
# info_alpha_beta = my_behavior.data_generation_alpha_beta(x0_vec[:, 1], p, N_horizon_test, 0.01)

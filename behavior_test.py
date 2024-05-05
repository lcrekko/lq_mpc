import numpy as np
import control as ct
from utils import local_radius, circle_generator, default_color_generator
# from scipy.linalg import cho_factor
# import numpy.linalg as la
import math
from utils_class import LQ_RDP_Behavior, Plotter_PF_LQMPC

# --------------Initialization---------------

# Specify the estimated system -- We use marginal stable
A = np.array([[1, 0.7], [0.12, 0.4]])
B = np.array([[1], [1.2]])

# Specify the cost -- We use the identity as the most simple case
Q = 2 * np.eye(2)
R = np.eye(1)

# Specify the input constraints -- We simply confine -1 <= u <= 1
F_u = np.array([[10], [-10]])

# Specify the number of the initial states per circle
N_points = 8

# Specify the rotation angles for rotating the initial state
# my_theta = np.linspace(0, math.pi, N_init)

# ---------------Prerequisite-----------------

# Computing the LQR optimal gain -- Remark: the convention is A - BK
K_lqr, P_lqr, eig_lqr = ct.dlqr(A, B, Q, R)

# Computing the terminal set for the LQR gain
epsilon_lqr = local_radius(F_u, -K_lqr, Q)

# Specifying the extension factor -- Remark: this factor must be greater than 1
ratio_x0 = 1.5

# ------------- Computing the energy bar M_{\hat{V}} -------------
# A test horizon
N_horizon = 6

# Defining reference trajectory
x_ref = np.zeros((2, N_horizon))  # Example reference trajectory
u_ref = np.zeros((1, N_horizon))  # Example reference control inputs

# Specify the modeling error
err_level = 0.01
err_nominal = {'e_A': err_level, 'e_B': err_level}

# Specify the (q.q) pair
p = np.array([0.1, 1, 0.6])

# Create the test RDP behavior class
N_min = 6
N_max = 10
e_power_min = -6
e_power_max = -2
my_behavior = LQ_RDP_Behavior(A, B, Q, R, F_u, -K_lqr, N_min, N_max, e_power_min, e_power_max)

# Compute the energy bound
M_V = my_behavior.OL_energy_bound(N_horizon, N_points, ratio_x0, x_ref, u_ref)
# Output the considered initial state for creating the coefficient trajectory
x0_vec = circle_generator(N_points, ratio_x0, epsilon_lqr, Q)

# ------------------- Generate data of the error-consistent functions --------------------
# data curve
data_xi = my_behavior.data_generation_xi(-K_lqr, M_V, N_horizon, err_nominal)
data_alpha, data_beta = my_behavior.data_generation_alpha_beta(x0_vec[:, 1], p, N_horizon, err_nominal)

# print(f"The xi data is", info_xi)
# print(f"The alpha data is", info_alpha)
# print(f"The beta data is", info_beta)

# Specify the true system
A_true = np.array([[1.01, 0.7], [0.12, 0.41]])
B_true = np.array([[1], [1.21]])
sys_true = {'A_true': A_true, 'B_true': B_true}

# Specify the simulation information
sim_info = {'T_mpc': 20, 'N_opc': 20}

# Integrate the reference trajectory
x_ref_long = np.zeros((2, sim_info['N_opc']))  # Example reference trajectory
u_ref_long = np.zeros((1, sim_info['N_opc']))  # Example reference control inputs
info_ref = {'x_ref': x_ref, 'u_ref': u_ref, 'x_ref_long': x_ref_long, 'u_ref_long': u_ref_long}

# --------------------- Generate the data for plotting the performance plane ----------------------
# Specify the radius vector
incremental_radius = 0.1
ratio_vec = np.arange(1 + incremental_radius, ratio_x0 + incremental_radius, incremental_radius)

# test plane data
# data_surface = my_behavior.data_generation_plane(N_horizon, sim_info, sys_true, err_nominal, info_ref,
# M_V, p, N_points, ratio_vec)

# plane data (mesh format)
quadrant_range = np.array([0.12, 0.16])
data_surface = my_behavior.data_generation_mesh(N_horizon, sim_info, sys_true, err_nominal, info_ref,
                                                M_V, p, quadrant_range)

# print(f"The plane data is", info_plane)

# --------------------- Plotting the data ----------------------
# Define the plotter
my_behavior_plotter = Plotter_PF_LQMPC(data_surface, data_xi, data_alpha, data_beta,
                                       N_min, N_max, e_power_min, e_power_max)

color_dict = default_color_generator()

# Specify figure setting
fig_height_rectangle = 4
fig_height_square = 3
fig_size_rectangle = np.array([fig_height_rectangle * 0.5 * (math.sqrt(5) + 1), fig_height_rectangle])
fig_size_square = np.array([fig_height_square * 0.5 * (math.sqrt(5) + 1), fig_height_square * 0.5 * (math.sqrt(5) + 1)])

my_font_type = "Times New Roman"
my_font_size = {"title": fig_height_square * 4, "label": fig_height_square * 4, "legend": fig_height_square * 4}

my_behavior_plotter.plot_plane_comparison(fig_size_square, my_font_type, my_font_size, color_dict)
my_behavior_plotter.plot_fc_ec(fig_size_rectangle, my_font_type, my_font_size, err_level, N_horizon)

"""
This is the script for testing the classes LQ_RDP_Behavior_Multiple and Plotter_PF_LQMPC_Multiple,
figures will be saved in the same folder
"""
import numpy as np
import math
# from utils import error_matrix_generator (uncomment this line if you do NOT want to use preload data)
from utils import default_color_generator
from utils_class import LQ_RDP_Behavior_Multiple, Plotter_PF_LQMPC_Multiple

# ---------------------- Optimal control information -----------------------
# Specify the true system, we have an unstable system (the challenging case)
A = np.array([[1, 0.7], [0.12, 0.4]])
B = np.array([[1], [1.2]])

# Get the dimension
n_x = A.shape[0]
n_u = B.shape[1]

# Specify the cost
Q = 2 * np.eye(n_x)
R = np.eye(n_u)

# Specify the input constraints
F_u = np.vstack((10 * np.eye(n_u), -10 * np.eye(n_u)))

info_opc = {'A': A, 'B': B,
            'Q': Q, 'R': R,
            'F_u': F_u}

# --------------- Prediction horizon and reference information -------------
N_min = 6  # Minimum considered prediction horizon
N_max = 10  # Maximum considered prediction horizon
N_nominal = 7  # Nominal prediction horizon
N_opc = 30  # Open-loop control horizon
N_mpc = 30  # Closed-loop control horizon

info_N = {'N_min': N_min, 'N_max': N_max,
          'N_nominal': N_nominal, 'N_opc': N_opc,
          'N_mpc': N_mpc}

# Specify the reference
x_ref = np.zeros([n_x, N_nominal])  # mpc reference trajectory
u_ref = np.zeros([n_u, N_nominal])  # mpc reference control inputs
x_ref_long = np.zeros([n_x, N_opc])  # opc reference trajectory
u_ref_long = np.zeros([n_u, N_opc])  # opc reference control inputs

info_ref = {'x_ref': x_ref, 'u_ref': u_ref,
            'x_ref_long': x_ref_long, 'u_ref_long': u_ref_long}

# --------------------- Modeling error information ---------------------
e_min = 10 ** (-3)
e_max = 10 ** (-2)
e_nominal = 5 * (10 ** (-3))

info_e = {'e_min': e_min, 'e_max': e_max,
          'e_nominal': e_nominal}
N_matrix = 20
'''
(Uncomment this section if you do NOT want to use the preload data)
# --------------------- Generate the error matrices ----------------------
error_vec = np.array([10 ** i for i in range(info_e_pow['e_pow_min'],
                                             info_e_pow['e_pow_max'] + 1)])

output = error_matrix_generator(A, B, error_vec, N_matrix, '2')
'''
# ----------------- Other parameters for computing the performance bound -----------------

# Specify the number of the initial states that will be simulated for determining the error bound M_V
N_points = 8

# Specify the extension factor -- Remark: this factor must be greater than 1
ext_radius_max = 1.5

# Specify the parameter p
p = np.array([0.1, 1, 0.6])

# ----------------- Parameters for plotting ------------------
# the color dictionary
color_dict = default_color_generator()

# figure size
fig_height_rectangle = 3
# using golden-ratio format
fig_size_rectangle = np.array([fig_height_rectangle * 0.5 * (math.sqrt(5) + 1), fig_height_rectangle])

# font type and size
my_font_type = "Times New Roman"
my_font_size_rectangle = {"title": fig_height_rectangle * 6, "label": fig_height_rectangle * 6,
                          "legend": fig_height_rectangle * 6, "tick": fig_height_rectangle * 6}
# saving information
info_save = {'type': 'svg', 'dpi': 1000}

# ----------------- Main commands -----------------
'''
# (Uncomment this section if you want to recompute the data again)
# initialize the class for generating data
my_behavior_multiple = LQ_RDP_Behavior_Multiple(info_opc, info_N, info_e,
                                                N_matrix, 'f')
# generate data
data_table = my_behavior_multiple.data_generation(N_points, ext_radius_max, info_ref, p)
'''
data_table = np.load('data_lq_mpc_multipleSys.npz')
# data_table = dict(data_table)

# initialize the class for plotting
my_plotter = Plotter_PF_LQMPC_Multiple(color_dict, fig_size_rectangle,
                                       my_font_type, my_font_size_rectangle, info_save)
# Plotting variation for modeling error
info_zoom_error = {'zoom': True,
                   'ratio': 3,
                   'loc': 'lower right',
                   'x_range': (0.004, 0.005),
                   'set_x_ticks': False,
                   'x_ticks': [0.004, 0.005],
                   'y_auto': True,
                   'y_range': (0, 1)}

my_plotter.plotter_error(N_nominal, data_table['error'],
                         data_table['alpha_table_error'], data_table['beta_table_error'],
                         data_table['xi_table_error'], data_table['bound_table_error'],
                         info_zoom_error, 'error_variation')

# Plotting variation for prediction horizon
info_zoom_horizon = {'zoom': True,
                     'ratio': 10,
                     'loc': 'lower right',
                     'x_range': (6.9, 7.1),
                     'set_x_ticks': True,
                     'x_ticks': [7],
                     'y_auto': True,
                     'y_range': (0, 1)}

my_plotter.plotter_horizon(e_nominal, data_table['horizon'],
                           data_table['alpha_table_horizon'], data_table['beta_table_horizon'],
                           data_table['xi_table_horizon'], data_table['bound_table_horizon'],
                           info_zoom_horizon, 'horizon_variation')

info_zoom_error_true_cost = {'zoom': False,
                             'ratio': 3,
                             'loc': 'lower right',
                             'x_range': (0.004, 0.005),
                             'set_x_ticks': False,
                             'x_ticks': [0.004, 0.005],
                             'y_auto': True,
                             'y_range': (0, 1)}
info_zoom_horizon_true_cost = {'zoom': False,
                               'ratio': 10,
                               'loc': 'lower right',
                               'x_range': (6.9, 7.1),
                               'set_x_ticks': True,
                               'x_ticks': [7],
                               'y_auto': True,
                               'y_range': (0, 1)}

my_plotter.plotter_true_cost(N_nominal, data_table['error'],
                             e_nominal, data_table['horizon'],
                             data_table['true_cost_error'], data_table['true_cost_horizon'],
                             info_zoom_error_true_cost, info_zoom_horizon_true_cost,
                             'true_cost_variation')

print('The spectral radius of matrix A is', np.max(np.abs(np.linalg.eigvals(A))))
print(data_table['V_expert'])

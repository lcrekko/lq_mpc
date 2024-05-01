# import cvxpy as cp
# import control as ct
# from utils import *
import math
from utils_class import LQ_MPC_Controller, LQ_MPC_Simulator, Plotter_MPC
import numpy as np

"""
Example usage of an open-loop MPC
"""

N_open = 20  # Prediction horizon

# Specify the estimated system -- We use marginal stable
A = np.array([[1, 0.7], [0.12, 0.4]])
B = np.array([[1], [1.2]])

# Get the dimension
n_x = A.shape[0]
n_u = B.shape[1]

# Specify the cost
Q = np.eye(n_x)
R = np.eye(n_u)

# Specify the initial condition
x0 = np.array([0.1125, 0.19])

# Specify the input constraints
F_u = np.vstack((10 * np.eye(n_u), -10 * np.eye(n_u)))

# Initialize the MPC controller
mpc_controller = LQ_MPC_Controller(N_open, A, B, Q, R, Q, F_u)

# Set the tracking trajectory to be 0 identically, the task is simply a stabilization problem
x_ref = np.zeros((n_x, N_open))  # Example reference trajectory
u_ref = np.zeros((n_u, N_open))  # Example reference control inputs

# Solve the MPC controller
MPC_info = mpc_controller.solve(x0, x_ref, u_ref)

# Print the results
print(f"Open-loop optimal value: {MPC_info['V_N']}")
print(f"First applied input: {MPC_info['u_0']}")


"""
Example usage of an closed-loop MPC
"""
# Simulation length
T = 20
N_closed = 6

# Specify the true system
A_true = np.array([[1.01, 0.7], [0.12, 0.41]])
B_true = np.array([[1], [1.21]])

# Define the MPC simulator using the nominal model
mpc_simulator = LQ_MPC_Simulator(T, N_closed, A, B, Q, R, Q, F_u)

# Simulate the trajectory using the true model
MPC_traj_info = mpc_simulator.simulate(x0, A_true, B_true, x_ref, u_ref)

# Print the results
print(f"Closed-loop cost: {MPC_traj_info['J_T']}")
print(f"The last state: {MPC_traj_info['X'][:, -1]}")


"""
Plotting the closed-loop trajectory
"""

# Specify figure setting
fig_height_rectangle = 4
fig_height_square = 3
fig_size_rectangle = np.array([fig_height_rectangle * 0.5 * (math.sqrt(5) + 1), fig_height_rectangle])
fig_size_square = np.array([fig_height_square * 0.5 * (math.sqrt(5) + 1), fig_height_square * 0.5 * (math.sqrt(5) + 1)])

my_font_type = "Times New Roman"
my_font_size = {"title": fig_height_square * 4, "label": fig_height_square * 4, "legend": fig_height_square * 4}

# do the plotting
my_plot = Plotter_MPC(MPC_traj_info['X'], MPC_traj_info['U'])
my_plot.plot_1_D_state(fig_size_rectangle, my_font_type, my_font_size)
my_plot.plot_2_D_state(fig_size_square, my_font_type, my_font_size)

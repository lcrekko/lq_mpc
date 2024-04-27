# import cvxpy as cp
# import control as ct
# from utils import *
from utils_class import LQ_MPC_Controller, LQ_MPC_Simulator
import numpy as np

# ----------- Example usage of an open-loop MPC ----------
N = 10  # Prediction horizon

# Specify the system
A = np.array([[1, 0.15], [0.1, 1]])
B = np.array([[0.1], [1.1]])

# Get the dimension
n_x = A.shape[0]
n_u = B.shape[1]

# Specify the cost
Q = np.eye(n_x)
R = np.eye(n_u)

# Specify the initial condition
x0 = np.array([1, 1])

# Specify the input constraints
F_u = np.vstack((np.eye(n_u), -np.eye(n_u)))

# Initialize the MPC controller
mpc_controller = LQ_MPC_Controller(N, A, B, Q, R, Q, F_u)

# Set the tracking trajectory to be 0 identically, the task is simply a stabilization problem
x_ref = np.zeros((n_x, N))  # Example reference trajectory
u_ref = np.zeros((n_u, N))  # Example reference control inputs

# Solve the MPC controller
MPC_info = mpc_controller.solve(x0, x_ref, u_ref)

# Print the results
print(f"Optimal value: {MPC_info['V_N']}")
print(f"First applied input: {MPC_info['u_0']}")

# This is a test

"""
# ----------- Example usage of a closed-loop MPC ----------

# Simulation length
T = 30

# Specify the true system
A_true = np.array([[1, 0.05], [0.0, 1]])
B_true = np.array([[0], [1.1]])

# Define the MPC simulator using the nominal model
mpc_simulator = LQ_MPC_Simulator(T, N, A, B, Q, R, Q, F_u)

# Simulate the trajectory using the true model
MPC_traj_info = mpc_simulator.simulate(x0, A_true, B_true, x_ref, u_ref)

# Print the results
print("Closed-loop Cost:", MPC_traj_info['J_T'])
"""
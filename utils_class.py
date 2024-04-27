import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils import bar_u_solve, fc_ec_E, local_radius, ex_stability_lq, ex_stability_bounds, fc_omega_eta, fc_ec_h
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman"
})  # This command is really useful ensuring the math font to be computer modern


class LQ_MPC_Controller:
    """
    This class is responsible for the open-loop linear quadratic MPC solver
    """

    def __init__(self, N, A, B, Q, R, P, F_u):
        """
        Initialize the class
        :param N: prediction horizon
        :param A: matrix A
        :param B: matrix B
        :param Q: matrix Q
        :param R: matrix R
        :param P: matrix P (terminal cost)
        :param F_u: the constraints for the input
        """
        self.N = N
        # self.dt = dt  # Time step

        # System matrices
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.F_u = F_u

        # Define optimization variables
        self.u = cp.Variable((B.shape[1], N))

    def solve(self, x0, x_ref, u_ref):
        """
        Solve the MPC problem
        :param x0: initial value of the x
        :param x_ref: reference state
        :param u_ref: reference input
        :return: a dictionary containing the input and the value function of the MPC
                 WARNING: The output value of the first step input has NO dimension, this should be
                 taken into account when building the closed-loop MPC simulator!!
        """
        # Define optimization problem
        cost = 0.0
        constraints = []
        x = x0
        for i in range(self.N):
            # System dynamics constraint
            x = self.A @ x + self.B @ self.u[:, i]

            # Tracking state cost
            if i < self.N - 1:
                # adding the cost for k = 1 to k = N-1
                cost += cp.quad_form(x - x_ref[:, i], self.Q)
            else:
                # adding the terminal cost for k = N
                cost += cp.quad_form(x - x_ref[:, i], self.P)

            # Tracking input cost
            cost += cp.quad_form(self.u[:, i] - u_ref[:, i], self.R)

            # State constraints - None

            # Input constraints
            # constraints += [self.F_u @ self.u[:, i] <= np.ones((self.F_u.shape[0], 1))]
            constraints += [self.F_u @ self.u[:, i] <= 1]

        # Create optimization objective and problem
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective, constraints)

        # Solve optimization problem
        problem.solve()

        # Return a dictionary that contains the first input as well as the open-loop MPC value function
        return {'u_0': self.u[:, 0].value, 'V_N': objective.value + x0 @ self.Q @ x0}


class Plotter_MPC:
    """
    This class is used to visualize the MPC trajectories, both state and input
    """

    def __init__(self, X, U):
        """
        The initialization function for the Plotter_MPC class
        :param X: The state trajectory
        :param U: The input trajectory
        """
        self.X = X
        self.U = U
        self.n_x = X.shape[0]
        self.n_u = U.shape[0]
        self.T = X.shape[1]

    def plot_1_D_state(self, size, font_type, font_size):
        """
        Plot the state trajectory in one dimension as time-dependent curves
        :param size: size of the figure
        :param font_type: font type in the figure
        :param font_size: different font size for the text (e.g., title, legend, and labels)
        :return: NONE
        """
        # Plotting each of the trajectories
        plt.figure(figsize=size)
        for i in range(self.n_x):
            plt.plot(self.X[i, :],
                     drawstyle='steps-post',  # for plotting in zero-order holder style
                     label=r'$x_{}$'.format(i + 1))

        # Specify the labels
        plt.xlabel(r'Time Step ($t$)', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.ylabel(r'Value of States ($x_i$)',
                   fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.title('State Trajectory', fontdict={'family': font_type, 'size': font_size["title"], 'weight': 'bold'})
        plt.legend(fontsize=font_size["legend"], prop={'family': font_type, 'size': font_size["legend"]})
        # Add Grid
        plt.grid(True)
        # Save the plot
        plt.savefig('1-D_state.svg', format='svg', dpi=300)
        # Show the plot
        plt.show()

    def plot_1_D_input(self, size, font_type, font_size):
        """
        Plot the input trajectory in one dimension as time-dependent curves
        :param size: size of the figure
        :param font_type: font type in the figure
        :param font_size: different font size for the text (e.g., title, legend, and labels)
        :return: NONE
        """
        # Plotting each of the trajectories
        plt.figure(figsize=size)
        for i in range(self.n_u):
            plt.plot(self.U[i, :],
                     drawstyle='steps-post',
                     label=r'$u_{}$'.format(i + 1))

        # Specify the descriptions
        plt.xlabel(r'Time Step ($t$)', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.ylabel(r'Value of Inputs ($u_i$)',
                   fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.title('Input Trajectory', fontdict={'family': font_type, 'size': font_size["title"], 'weight': 'bold'})
        plt.legend(fontsize=font_size["legend"], prop={'family': font_type, 'size': font_size["legend"]})
        # Add Grid
        plt.grid(True)
        # Save the plot
        plt.savefig('1-D_input.svg', format='svg', dpi=300)
        # Show the plot
        plt.show()

    def plot_2_D_state(self, size, font_type, font_size):
        """
        Plot the state trajectory in two dimensions
        WARNING: ONLY use if the state is of dimension 2
        :param size:
        :param font_type:
        :param font_size:
        :return:
        """
        # Self checking the dimension of the state
        if self.n_x >= 3:
            raise ValueError("The dimension of the state must be 2.")

        # get the two dimensions as separate coordinates
        x_coordinate = self.X[0, :]
        y_coordinate = self.X[1, :]

        # Do the plotting
        plt.figure(figsize=size)
        # First plot all the data points
        plt.plot(x_coordinate, y_coordinate,
                 marker='o',
                 linestyle='-',
                 markerfacecolor='r', markeredgecolor='b', markersize=10,
                 label=r'$x = (x_1, x_2)$')
        # Distinguish the initial state
        plt.plot(x_coordinate[0], y_coordinate[0],
                 marker='s', markersize=10, markerfacecolor='y', markeredgecolor='m',
                 label=r'$x(0)$')
        # Set equal axis
        plt.axis('equal')

        # Specify the descriptions
        plt.xlabel(r'$x_1$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.ylabel(r'$x_2$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        plt.title('State Trajectory (2-D)',
                  fontdict={'family': font_type, 'size': font_size["title"], 'weight': 'bold'})
        plt.legend(fontsize=font_size["legend"], prop={'family': font_type, 'size': font_size["legend"]})
        # Add Grid
        plt.grid(True)
        # Save the plot
        plt.savefig('2-D_state.svg', format='svg', dpi=300)
        # Show the plot
        plt.show()


class LQ_MPC_Simulator:
    """
    This class is used to solve the closed-loop MPC problem using the open-loop MPC
    """

    def __init__(self, T, N, A, B, Q, R, P, F_u):
        """
        The initialization of the LQ_MPC_Simulator
        :param T: Simulation horizon
        :param N: Prediction horizon
        :param A: matrix A
        :param B: matrix B
        :param Q: matrix Q
        :param R: matrix R
        :param P: matrix P (terminal cost)
        :param F_u: matrix in the input constraints
        """
        self.T = T
        self.N = N
        # self.dt = dt  # Time step

        # System matrices
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P
        self.F_u = F_u

        self.U = np.zeros((B.shape[1], T))
        self.X = np.zeros((A.shape[1], T + 1))

    def simulate(self, x0, A_true, B_true, x_ref, u_ref):
        """
        Simulate the closed-loop MPC problem
        :param x0: initial condition
        :param A_true: matrix A_es (estimated system)
        :param B_true: matrix B_es (estimated system)
        :param x_ref: reference state
        :param u_ref: reference input
        :return: a dictionary containing the closed-loop state and input trajectory as well as the total cost
        """

        # initialize the first state
        # WARNING: for convenience, the x_0 is specified without dimension, so here we need to reshape it first
        self.X[:, 0:1] = x0.reshape(self.A.shape[1], 1)

        # initialize the cost
        cost = self.X[:, 0] @ self.Q @ self.X[:, 0]

        # define our MPC controller
        myMPC = LQ_MPC_Controller(self.N, self.A, self.B, self.Q, self.R, self.P, self.F_u)

        for i in range(self.T):
            # solve the MPC to get the control input
            # WARNING: here we direct extract X[:,i] to gets a no dimension vector
            temp_out = myMPC.solve(self.X[:, i], x_ref, u_ref)

            # store the input
            # WARNING: The returned u_0 is of NO dimension, and we need to reshape it!!
            self.U[:, i:i + 1] = temp_out['u_0'].reshape(self.B.shape[1], 1)

            # update the state
            # WARNING: this computed x_next has NO dimension, so we still reshape it first before assigning
            x_next = A_true @ self.X[:, i] + B_true @ temp_out['u_0']

            self.X[:, i + 1:i + 2] = x_next.reshape(self.A.shape[1], 1)

            # update the cost
            cost += x_next @ self.Q @ x_next  # add the next state cost
            cost += temp_out['u_0'] @ self.R @ temp_out['u_0']  # add the input cost

        return {'X': self.X, 'U': self.U, 'J_T': cost}


class LQ_RDP_Calculator:
    """
    This class is used to compute the coefficient of RDP performance bound
    """

    def __init__(self, N, A, B, Q, R, F_u, e_A, e_B):
        """
        This is the initialization of the class
        :param N: Prediction horizon
        :param A: matrix A, estimated
        :param B: matrix B, estimated
        :param Q: matrix Q
        :param R: matrix R
        :param F_u: input constraints
        :param e_A: mismatch in A
        :param e_B: mismatch in B
        """
        self.N = N
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F_u = F_u
        self.e_A = e_A
        self.e_B = e_B

    def energy_bound(self, x, p):
        """
        This function returns the value of alpha and beta in the energy bounding relation
        :param x: initial state
        :param p: coefficients of the linear bound for quadratic terms
        :return: a dictionary that contains the value of alpha and beta in the energy bounding relation
        """

        # compute the maximum value of \|u\|^2_2 under the input constraints
        bar_u = bar_u_solve(self.F_u)

        # compute the values of E_\psi, E_u and E_{\psi,u}
        val_E = fc_ec_E(self.N, self.e_A, self.e_B, self.A, self.B, self.Q, self.R, x, bar_u)

        # compute the value of q given p
        q = 1 / p

        # compute the term alpha
        term_alpha_1 = (p[0] * math.sqrt(val_E['E_psi']) + p[1] * math.sqrt(val_E['E_psi_u']) +
                        p[0] * math.sqrt(val_E['E_psi']) * p[1] * math.sqrt(val_E['E_psi_u']))
        term_alpha_2 = p[2] * math.sqrt(val_E['E_u'])
        my_alpha = np.max([term_alpha_1, term_alpha_2])

        # compute the term beta
        term_beta_1 = (1 + p[1] * math.sqrt(val_E['E_psi'])) * (q[1] * math.sqrt(val_E['E_psi_u']) + val_E['E_psi_u'])
        term_beta_2 = q[2] * math.sqrt(val_E['E_u']) + val_E['E_u'] + q[0] * math.sqrt(val_E['E_psi']) + val_E['E_psi']
        my_beta = term_beta_1 + term_beta_2

        return {'alpha': my_alpha, 'beta': my_beta}

    def energy_decreasing(self, hatK, K, M_V):
        """
        Compute the term xi and eta in the RDP energy-decreasing inequality
        :param hatK: The terminal linear gain for the deviated state
        :param K: The terminal linear gain for the original state using the estimated system
        :param M_V: The bound of the open-loop MPC value function
        :return: a dictionary that contains the values of the term xi and eta
        """

        # compute the local radius
        my_epsilon = local_radius(self.F_u, K, self.Q)

        # compute the stability information to get a desired gamma
        info_stable = ex_stability_lq(self.A, self.B, self.Q, self.R, K)

        # compute the bounding constants
        info_bd = ex_stability_bounds(info_stable['gamma'], my_epsilon, M_V)

        # compute the terms omega and eta
        info_omega_eta = fc_omega_eta(self.N, self.A, self.B, self.Q, self.R, K, hatK, info_bd['L_V'], info_bd['N_0'])

        # compute the term h
        my_h = fc_ec_h(self.e_A, self.e_B, self.Q, self.R)

        # compute the term xi
        my_xi = my_h * info_omega_eta['omega_N1'] + 2 * math.sqrt(my_h) * info_omega_eta['omega_N0d5']

        return {'xi': my_xi, 'eta': info_omega_eta['eta']}

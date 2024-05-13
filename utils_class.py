import math
import cvxpy as cp
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from utils import (bar_u_solve, bar_d_u_solve,
                   local_radius, ex_stability_lq, ex_stability_bounds,
                   fc_omega_eta, fc_ec_h, fc_ec_E, fc_omega_eta_extension,
                   circle_generator, gradient_color,
                   error_matrix_generator, statistical_continuous)

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

    def __init__(self, A, B, Q, R, F_u):
        """
        This is the initialization of the class
        :param A: matrix A, estimated
        :param B: matrix B, estimated
        :param Q: matrix Q
        :param R: matrix R
        :param F_u: input constraints
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F_u = F_u

    def energy_bound(self, N, e_A, e_B, x, p):
        """
        This function returns the value of alpha and beta in the energy bounding relation
        :param N: prediction horizon
        :param e_A: error in matrix A
        :param e_B: error in matrix B
        :param x: initial state
        :param p: coefficients of the linear bound for quadratic terms
        :return: a dictionary that contains the value of alpha and beta in the energy bounding relation
        """

        # compute the maximum value of \|u\|^2_2 under the input constraints
        bar_u = bar_u_solve(self.F_u)

        # compute the maximum value of \|u_1 - u_2\|^2_2 under the input constraints
        bar_d_u = bar_d_u_solve(self.F_u)

        # compute the values of E_\psi, E_u and E_{\psi,u}
        val_E = fc_ec_E(N, e_A, e_B, self.A, self.B, self.Q, self.R, x, bar_u, bar_d_u)

        # compute the value of q given p
        q = 1 / p

        # compute the term alpha
        term_alpha_1 = (p[0] * math.sqrt(val_E['E_psi']) + p[2] * math.sqrt(val_E['E_psi_u']) +
                        p[0] * math.sqrt(val_E['E_psi']) * p[2] * math.sqrt(val_E['E_psi_u']))
        term_alpha_2 = p[1] * math.sqrt(val_E['E_u'])
        my_alpha = np.max([term_alpha_1, term_alpha_2])

        # compute the term beta
        term_beta_1 = (1 + p[0] * math.sqrt(val_E['E_psi'])) * (q[2] * math.sqrt(val_E['E_psi_u']) + val_E['E_psi_u'])
        term_beta_2 = q[1] * math.sqrt(val_E['E_u']) + val_E['E_u'] + q[0] * math.sqrt(val_E['E_psi']) + val_E['E_psi']
        my_beta = term_beta_1 + term_beta_2

        return {'alpha': my_alpha, 'beta': my_beta}

    def energy_decreasing(self, N, e_A, e_B, K, M_V):
        """
        Compute the term xi and eta in the RDP energy-decreasing inequality
        :param N: prediction horizon
        :param e_A: error in matrix A
        :param e_B: error in matrix B
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
        info_omega_eta = fc_omega_eta(N, self.A, self.B, self.Q, self.R, K, info_bd['L_V'], info_bd['N_0'])

        # compute the term h
        my_h = fc_ec_h(e_A, e_B, self.Q, self.R)

        # compute the term xi
        my_xi = my_h * info_omega_eta['omega_N1'] + 2 * math.sqrt(my_h) * info_omega_eta['omega_N0d5']

        return {'xi': my_xi, 'eta': info_omega_eta['eta']}

    def energy_decreasing_extension(self, N, e_A, e_B, K, hatK, M_V):
        """
        Compute the term xi and eta in the RDP energy-decreasing inequality
        :param N: prediction horizon
        :param e_A: error in matrix A
        :param e_B: error in matrix B
        :param K: The terminal linear gain for the original state using the estimated system
        :param hatK: The terminal linear gain for the deviated system
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
        info_omega_eta = fc_omega_eta_extension(N, self.A, self.B, self.Q, self.R, K, hatK,
                                                info_bd['L_V'], info_bd['N_0'])

        # compute the term h
        my_h = fc_ec_h(e_A, e_B, self.Q, self.R)

        # compute the term xi
        my_xi = my_h * info_omega_eta['omega_N1'] + 2 * math.sqrt(my_h) * info_omega_eta['omega_N0d5']

        return {'xi': my_xi, 'eta': info_omega_eta['eta']}


class LQ_RDP_Behavior:
    """
    This class is used to compute the coefficient of RDP performance with varying horizon and modeling error
    """

    def __init__(self, A, B, Q, R, F_u, K, N_min, N_max, e_pow_min, e_pow_max):
        """
        This is the initialization of the class
        :param A: matrix A, estimated
        :param B: matrix B, estimated
        :param Q: matrix Q
        :param R: matrix R
        :param F_u: input constraints
        :param K: the used feedback gain
        :param N_min: minimum used horizon
        :param N_max: maximum used horizon
        :param e_pow_min: minimum used power of the error
        :param e_pow_max: maximum used power of the error
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.F_u = F_u
        self.K = K
        self.epsilon = local_radius(F_u, K, Q)
        self.calculator = LQ_RDP_Calculator(A, B, Q, R, F_u)
        self.horizon = np.arange(N_min, N_max + 1)
        self.error_vec = np.array([10 ** i for i in range(e_pow_min, e_pow_max + 1)])

    def OL_energy_bound(self, N, N_points, ext_radius_max, x_ref, u_ref):
        """
        Calculates the maximum energy M_V for performance computation
        :param N: prediction horizon
        :param N_points: The number of data points considered on the circle
        :param ext_radius_max: the ratio of radius extension
        :param x_ref: the state reference
        :param u_ref: the input reference
        :return: the maximum open-loop energy bound
        """
        # get the circle
        x0_vec = circle_generator(N_points, ext_radius_max, self.epsilon, self.Q)

        # Specify the MPC
        my_MPC = LQ_MPC_Controller(N, self.A, self.B, self.Q, self.R, self.Q, self.F_u)

        # Initialize a variable to store the open-loop cost
        energy_vec = np.zeros(x0_vec.shape[1])

        # loop computation
        for i in range(x0_vec.shape[1]):
            info_MPC_test = my_MPC.solve(x0_vec[:, i], x_ref, u_ref)
            energy_vec[i] = info_MPC_test['V_N']

        # taking the maximum to get an estimate of the energy bar
        M_V = np.max(energy_vec)

        return M_V

    def data_generation_xi(self, K, M_V, N_nominal, err_nominal):
        """
        This is the function that generates a bunch of xi
        :param K: The terminal control law
        :param M_V: The energy bound
        :param N_nominal: The nominal prediction horizon, usually choose the minium one
        :param err_nominal: A dictionary that contains the nominal permitted error
        :return: a dictionary with xi data
        """
        xi_error = np.zeros(len(self.error_vec))
        for i in range(len(self.error_vec)):
            xi_error[i] = self.calculator.energy_decreasing(N_nominal, self.error_vec[i], self.error_vec[i], K, M_V)[
                'xi']

        # get the alpha and beta value
        my_e_A = err_nominal['e_A']
        my_e_B = err_nominal['e_B']

        xi_horizon = np.zeros(len(self.horizon))
        for i in range(len(self.horizon)):
            xi_horizon[i] = self.calculator.energy_decreasing(self.horizon[i], my_e_A, my_e_B, K, M_V)['xi']

        return {'error': xi_error, 'horizon': xi_horizon}

    def data_generation_alpha_beta(self, x, p, N_nominal, err_nominal):
        """
        This function generates a bunch of alpha and beta
        :param x: the evaluated starting point
        :param p: the value of scalars p
        :param N_nominal: The nominal prediction horizon, usually choose the minium one
        :param err_nominal: The nominal permitted error
        :return: two dictionaries with alpha and beta
        """
        alpha_error = np.zeros(len(self.error_vec))
        beta_error = np.zeros(len(self.error_vec))
        for i in range(len(self.error_vec)):
            alpha_error[i] = self.calculator.energy_bound(N_nominal, self.error_vec[i],
                                                          self.error_vec[i], x, p)['alpha']
            beta_error[i] = self.calculator.energy_bound(N_nominal, self.error_vec[i],
                                                         self.error_vec[i], x, p)['beta']

        # get the alpha and beta value
        my_e_A = err_nominal['e_A']
        my_e_B = err_nominal['e_B']

        alpha_horizon = np.zeros(len(self.horizon))
        beta_horizon = np.zeros(len(self.horizon))
        for i in range(len(self.error_vec)):
            alpha_horizon[i] = self.calculator.energy_bound(self.horizon[i], my_e_A,
                                                            my_e_B, x, p)['alpha']
            beta_horizon[i] = self.calculator.energy_bound(self.horizon[i], my_e_A,
                                                           my_e_B, x, p)['beta']

        return {'error': alpha_error, 'horizon': alpha_horizon}, {'error': beta_error, 'horizon': beta_horizon}

    def data_generation_plane(self, N, sim_info, sys_true, err_nominal, info_ref,
                              M_V, p, N_points, ratio_ext_radius):
        """
        This function generates the data for J_MPC, J_MPC_bound, and V_OPC value for each of the points on
        the circle
        :param N: Prediction horizon
        :param sim_info: A dictionary that contains the open-loop simulation horizon and closed-loop simulation horizon
        :param sys_true: A dictionary that contains the information of the true system
        :param err_nominal: A dictionary that contains the nominal error of the model
        :param info_ref: A dictionary that contains the reference of the state and input
        :param M_V: open-loop energy bound
        :param p: the scalars p
        :param N_points: The number of data points on the circle
        :param ratio_ext_radius: the ratio of the extra radius extension
        :return: A dictionary with all the data points and their corresponding cost values
        """
        # get the alpha and beta value
        e_A = err_nominal['e_A']
        e_B = err_nominal['e_B']

        # get the true system
        A_true = sys_true['A_true']
        B_true = sys_true['B_true']

        # get the simulation info
        T_mpc = sim_info['T_mpc']
        N_opc = sim_info['N_opc']

        # get the reference
        x_ref = info_ref['x_ref']
        u_ref = info_ref['u_ref']
        x_ref_long = info_ref['x_ref_long']
        u_ref_long = info_ref['u_ref_long']

        # Computing the xi and eta from the energy-decreasing module
        info_decrease = self.calculator.energy_decreasing(N, e_A, e_B, self.K, M_V)
        decreasing_factor = 1 / (1 - info_decrease['xi'] - info_decrease['eta'])

        # define the open-loop MPC
        ol_mpc = LQ_MPC_Controller(N_opc, A_true, B_true, self.Q, self.R, self.Q, self.F_u)

        # define the closed-loop MPC
        cl_mpc = LQ_MPC_Simulator(T_mpc, N, self.A, self.B, self.Q, self.R, self.Q, self.F_u)

        # ----------------- Computation of the cost values -----------------
        X_1 = []
        X_2 = []
        J_MPC_true = []
        J_MPC_bound = []
        V_OPC = []
        temp_alpha = np.zeros(N_points)  # initialization alpha
        temp_beta = np.zeros(N_points)  # initialization beta
        temp_J_MPC_true = np.zeros(N_points)  # initialization J_MPC
        temp_J_MPC_bound = np.zeros(N_points)  # initialization J_MPC_bound
        temp_V_OPC = np.zeros(N_points)  # initialization V_OPC

        for j in range(len(ratio_ext_radius)):
            # Computing the circle
            X_points = circle_generator(N_points, ratio_ext_radius[j], self.epsilon, self.Q)

            # The for loop (looping each of the points)
            for i in range(N_points):
                # Computing alpha and beta for each of the points on the circle
                temp_info_energy_bound = self.calculator.energy_bound(N, e_A, e_B, X_points[:, i], p)
                temp_alpha[i] = temp_info_energy_bound['alpha']
                temp_beta[i] = temp_info_energy_bound['beta']

                # Computing the open-loop MPC cost for each of the points, knowing the true system
                temp_info_ol_mpc = ol_mpc.solve(X_points[:, i], x_ref_long, u_ref_long)
                temp_V_OPC[i] = temp_info_ol_mpc['V_N']

                # Computing the cost bound for each of the points
                temp_J_MPC_bound[i] = decreasing_factor * (temp_alpha[i] * temp_V_OPC[i] + temp_beta[i])

                # Computing the closed-loop MPC cost for each of the points, knowing the estimated system
                temp_info_cl_mpc = cl_mpc.simulate(X_points[:, i], A_true, B_true, x_ref, u_ref)
                temp_J_MPC_true[i] = temp_info_cl_mpc['J_T']

            X_1 = np.append(X_1, X_points[0, :])
            X_2 = np.append(X_2, X_points[1, :])
            J_MPC_true = np.append(J_MPC_true, temp_J_MPC_true)
            J_MPC_bound = np.append(J_MPC_true, temp_J_MPC_bound)
            V_OPC = np.append(J_MPC_true, temp_V_OPC)

        X = np.vstack((X_1, X_2))

        return {'X': X, 'J_MPC_true': J_MPC_true, 'J_MPC_bound': J_MPC_bound, 'V_OPC': V_OPC}

    def data_generation_mesh(self, N, sim_info, sys_true, err_nominal, info_ref,
                             M_V, p, quadrant_range):
        """
        Generates data points that form a mesh grid, which is more convenient for plotting the surface
        :param N: Prediction horizon
        :param sim_info: Simulation information
        :param sys_true: System information (true system)
        :param err_nominal: nominal error of the model
        :param info_ref: Reference of the state and input
        :param M_V: Energy bound
        :param p: scalars p
        :param quadrant_range: the range in the first quadrant
        :return: A dictionary with all the data points and their corresponding cost values
        """
        # get the alpha and beta value
        e_A = err_nominal['e_A']
        e_B = err_nominal['e_B']

        # get the true system
        A_true = sys_true['A_true']
        B_true = sys_true['B_true']

        # get the simulation info
        T_mpc = sim_info['T_mpc']
        N_opc = sim_info['N_opc']

        # get the reference
        x_ref = info_ref['x_ref']
        u_ref = info_ref['u_ref']
        x_ref_long = info_ref['x_ref_long']
        u_ref_long = info_ref['u_ref_long']

        # Computing the xi and eta from the energy-decreasing module
        info_decrease = self.calculator.energy_decreasing(N, e_A, e_B, self.K, M_V)
        decreasing_factor = 1 / (1 - info_decrease['xi'] - info_decrease['eta'])

        # define the open-loop MPC
        ol_mpc = LQ_MPC_Controller(N_opc, A_true, B_true, self.Q, self.R, self.Q, self.F_u)

        # define the closed-loop MPC
        cl_mpc = LQ_MPC_Simulator(T_mpc, N, self.A, self.B, self.Q, self.R, self.Q, self.F_u)

        # Create the meth
        X = np.hstack((-quadrant_range['x'], quadrant_range['x']))
        Y = np.hstack((-quadrant_range['y'], quadrant_range['y']))
        coord_X, coord_Y = np.meshgrid(X, Y)

        # ----------------- Computation of the cost values -----------------
        J_MPC_true = np.zeros([2 * len(quadrant_range['x']), 2 * len(quadrant_range['y'])])
        J_MPC_bound = np.zeros([2 * len(quadrant_range['x']), 2 * len(quadrant_range['y'])])
        V_OPC = np.zeros([2 * len(quadrant_range['x']), 2 * len(quadrant_range['y'])])

        for i in range(2 * len(quadrant_range['x'])):
            for j in range(2 * len(quadrant_range['y'])):
                # Computing alpha and beta for each of the points on the circle
                temp_point = np.array([X[i], Y[j]])
                temp_info_energy_bound = self.calculator.energy_bound(N, e_A, e_B, temp_point, p)
                temp_alpha = temp_info_energy_bound['alpha']
                temp_beta = temp_info_energy_bound['beta']

                # Computing the open-loop MPC cost for each of the points, knowing the true system
                temp_info_ol_mpc = ol_mpc.solve(temp_point, x_ref_long, u_ref_long)
                V_OPC[i, j] = temp_info_ol_mpc['V_N']

                # Computing the cost bound for each of the points
                J_MPC_bound[i, j] = decreasing_factor * (temp_alpha * V_OPC[i, j] + temp_beta)

                # Computing the closed-loop MPC cost for each of the points, knowing the estimated system
                temp_info_cl_mpc = cl_mpc.simulate(temp_point, A_true, B_true, x_ref, u_ref)
                J_MPC_true[i, j] = temp_info_cl_mpc['J_T']

        return {'X': coord_X, 'Y': coord_Y, 'J_MPC_true': J_MPC_true, 'J_MPC_bound': J_MPC_bound, 'V_OPC': V_OPC}


class LQ_RDP_Behavior_Multiple:
    """
    This class is an extension of the class LQ_RDP_Behavior where we create
    multiple sets of data for each of the estimated system
    """

    def __init__(self, info_opc: dict, info_N: dict, info_e_pow: dict,
                 N_sys: int, norm_type: str, errM_import=True):
        """
        This is the initialization of the class LQ_RDP_Behavior_Multiple
        :param info_opc: Dictionary with information about the optimal control problem
               It contains the following items:
               1. 'A' -> The system matrix A
               2. 'B' -> The system matrix B
               3. 'Q' -> The objective matrix Q
               4. 'R' -> The objective matrix R
               5. 'F_u' -> The constraint matrix F_u
        :param info_N: Dictionary with information about the prediction horizon
               It contains the following items:
               1. 'N_min' -> The minimum prediction horizon
               2. 'N_max' -> The maximum prediction horizon
               3. 'N_nominal' -> The nominal prediction horizon
               4. 'N_opc' -> The open-loop optimal control horizon
        :param info_e_pow: Dictionary with information about the modeling error
               It contains the following items:
               1. 'e_pow_min' -> The minimum power of the modeling error
               2. 'e_pow_max' -> The maximum power of the modeling error
               3. 'e_pow_nominal' -> The nominal power of the modeling error
        :param N_sys: Number of systems for a given level of modeling error
        :param norm_type: the type of the considered norm, either 'f' or '2'
        :param errM_import: whether to import the error matrix from existing data
        """
        # extract optimal control information
        self.A_true = info_opc['A']
        self.B_true = info_opc['B']
        self.Q = info_opc['Q']
        self.R = info_opc['R']
        self.F_u = info_opc['F_u']

        # get the number of system
        self.N_sys = 5 * N_sys  # check the function random_matrix in utils to see why it is 5

        # extract the prediction information
        self.N_min = info_N['N_min']
        self.N_max = info_N['N_max']
        self.N_nominal = info_N['N_nominal']
        self.N_opc = info_N['N_opc']

        # extract the error information
        self.e_min = info_e_pow['e_min']
        self.e_max = info_e_pow['e_max']
        self.e_nominal = info_e_pow['e_nominal']

        # get the prediction vector
        self.horizon = np.arange(info_N['N_min'], info_N['N_max'] + 1)

        # get the error vector
        # self.error_vec = np.array([10 ** i for i in range(info_e_pow['e_pow_min'], info_e_pow['e_pow_max'] + 1)])
        self.error_vec = np.linspace(self.e_min, self.e_max, 10)

        # obtain the delta matrices
        if errM_import:
            self.error_A = np.load('error_A' + '_' + norm_type + '.npy')
            self.error_B = np.load('error_B' + '_' + norm_type + '.npy')
        else:
            dic_err_M = error_matrix_generator(self.A_true, self.B_true, self.error_vec, N_sys, norm_type)
            self.error_A = dic_err_M['error_A']
            self.error_B = dic_err_M['error_B']

        # Compute the expert cost
        self.mpc_open = LQ_MPC_Controller(self.N_opc, self.A_true, self.B_true, self.Q, self.R,
                                          self.Q, self.F_u)

        # Computing the LQR optimal gain -- Remark: the convention is A - BK
        K_lqr, _, _ = ct.dlqr(self.A_true, self.B_true, self.Q, self.R)

        # Computing the terminal set for the LQR gain
        self.epsilon_lqr = local_radius(self.F_u, -K_lqr, self.Q)

    def data_generation(self, N_points: int, ext_radius_max: float,
                        info_ref: dict, p: np.ndarray) -> dict:
        """
        Generates data xi tables, alpha tables, beta tables, and performance tables
        :param N_points: The number of points considered on the circle
        :param ext_radius_max: The extended radius of the minimum circle
        :param info_ref: the reference information
        :param p: the pair of scalars p
        :return: A dictionary with all the tables as well as the variation range
        """
        # ----------------- Determination of x_0 and Computation of V_{\infty} ----------------
        # get the open-loop reference trajectory
        x_ref_opc = info_ref['x_ref_long']
        u_ref_opc = info_ref['u_ref_long']

        # determine our starting point uniformly used in this example
        x0_vec = circle_generator(N_points, ext_radius_max, self.epsilon_lqr, self.Q)
        x_start_global = x0_vec[:, 1]

        # get the expert cost
        V_expert = self.mpc_open.solve(x_start_global, x_ref_opc, u_ref_opc)['V_N']

        # --------------- computation of variation w.r.t. error ------------------
        # Initialize the table
        alpha_table_error = np.zeros([self.N_sys, len(self.error_vec)])
        beta_table_error = np.zeros([self.N_sys, len(self.error_vec)])
        xi_table_error = np.zeros([self.N_sys, len(self.error_vec)])
        bound_table_error = np.zeros([self.N_sys, len(self.error_vec)])

        # x_ref_nominal = info_ref['x_ref']
        # u_ref_nominal = info_ref['u_ref']

        # Outer-loop (looping error)
        for i in range(len(self.error_vec)):
            # get the error level
            my_err = self.error_vec[i]
            # Inner-loop (looping the systems)
            for j in range(self.N_sys):
                # obtain the perturbed matrices A and B
                A = self.A_true + self.error_A[:, :, j, i]
                B = self.B_true + self.error_B[:, :, j, i]

                # ------------- Determine the energy bound M_V --------------
                # Specify the MPC
                my_MPC = LQ_MPC_Controller(self.N_nominal, A, B, self.Q, self.R, self.Q, self.F_u)

                # Initialize a variable to store the open-loop cost
                temp_energy_vec = np.zeros(x0_vec.shape[1])

                # loop computation
                for k in range(x0_vec.shape[1]):
                    info_MPC_test = my_MPC.solve(x0_vec[:, k], info_ref['x_ref'], info_ref['u_ref'])
                    temp_energy_vec[k] = info_MPC_test['V_N']

                # taking the maximum to get an estimate of the energy bar
                M_V = np.max(temp_energy_vec)

                # ------------- Computation of the value xi ----------------
                # initialize the calculator
                temp_calculator = LQ_RDP_Calculator(A, B, self.Q, self.R, self.F_u)

                # compute the stabilizing gain
                my_K, _, _ = ct.dlqr(A, B, self.Q, self.R)

                # remember to pass the negative K for consistency
                temp_info_decrease = temp_calculator.energy_decreasing(self.N_nominal, my_err, my_err,
                                                                       -my_K, M_V)
                # get the xi value
                xi_table_error[j, i] = temp_info_decrease['xi']
                # get eta for computing the bound (used the next subsection)
                temp_eta = temp_info_decrease['eta']

                # ------------- Computation of the value alpha and beta --------------
                temp_info_bound = temp_calculator.energy_bound(self.N_nominal, my_err, my_err,
                                                               x_start_global, p)
                # compute alpha
                alpha_table_error[j, i] = temp_info_bound['alpha']
                # compute beta
                beta_table_error[j, i] = temp_info_bound['beta']
                # compute bound
                bound_table_error[j, i] = ((alpha_table_error[j, i] * V_expert + beta_table_error[j, i]) /
                                           (1 - xi_table_error[j, i] - temp_eta))

        # --------------- computation of variation w.r.t. horizon ------------------
        # Initialize the table
        alpha_table_horizon = np.zeros([self.N_sys, len(self.horizon)])
        beta_table_horizon = np.zeros([self.N_sys, len(self.horizon)])
        xi_table_horizon = np.zeros([self.N_sys, len(self.horizon)])
        bound_table_horizon = np.zeros([self.N_sys, len(self.horizon)])

        # get the nominal error
        err_horizon = self.e_nominal
        # get the index of the nominal error
        try:
            index_sys = np.where(self.horizon == err_horizon)[0]
        except IndexError:
            print(f"Warning: the nominal error is not in the error list")

        # set the default index to be the last one (the greatest error level)
        index_sys = 4
        # extract the set of matrices
        err_A_nominal = self.error_A[:, :, :, index_sys]
        err_B_nominal = self.error_B[:, :, :, index_sys]

        # Outer-loop (looping horizon)
        for i in range(len(self.horizon)):
            x_ref_temp = np.zeros([self.A_true.shape[1], self.horizon[i]])
            u_ref_temp = np.zeros([self.B_true.shape[1], self.horizon[i]])
            for j in range(self.N_sys):
                # obtain the perturbed matrices A and B
                A = self.A_true + err_A_nominal[:, :, j]
                B = self.B_true + err_B_nominal[:, :, j]

                # ------------- Determine the energy bound M_V --------------
                # Specify the MPC
                my_MPC = LQ_MPC_Controller(self.horizon[i], A, B, self.Q, self.R, self.Q, self.F_u)

                # Initialize a variable to store the open-loop cost
                temp_energy_vec = np.zeros(x0_vec.shape[1])

                # loop computation
                for k in range(x0_vec.shape[1]):
                    info_MPC_test = my_MPC.solve(x0_vec[:, k], x_ref_temp, u_ref_temp)
                    temp_energy_vec[k] = info_MPC_test['V_N']

                # taking the maximum to get an estimate of the energy bar
                M_V = np.max(temp_energy_vec)

                # ------------- Computation of the value xi ----------------
                # Initialize the calculator
                temp_calculator = LQ_RDP_Calculator(A, B, self.Q, self.R, self.F_u)

                # Compute the stabilizing gain
                my_K, _, _ = ct.dlqr(A, B, self.Q, self.R)

                # remember to pass the negative K for consistency
                temp_info_decrease = temp_calculator.energy_decreasing(self.horizon[i], err_horizon, err_horizon,
                                                                       -my_K, M_V)
                # get the xi value
                xi_table_horizon[j, i] = temp_info_decrease['xi']
                # get eta for computing the bound (used the next subsection)
                temp_eta = temp_info_decrease['eta']

                # ------------- Computation of the value alpha and beta --------------
                temp_info_bound = temp_calculator.energy_bound(self.horizon[i], err_horizon, err_horizon,
                                                               x_start_global, p)
                # compute alpha
                alpha_table_horizon[j, i] = temp_info_bound['alpha']
                # compute beta
                beta_table_horizon[j, i] = temp_info_bound['beta']
                # compute bound
                bound_table_horizon[j, i] = ((alpha_table_horizon[j, i] * V_expert + beta_table_horizon[j, i]) /
                                             (1 - xi_table_horizon[j, i] - temp_eta))

        out_dict = {'error': self.error_vec,
                    'horizon': self.horizon,
                    'alpha_table_error': alpha_table_error,
                    'beta_table_error': beta_table_error,
                    'xi_table_error': xi_table_error,
                    'bound_table_error': bound_table_error,
                    'alpha_table_horizon': alpha_table_horizon,
                    'beta_table_horizon': beta_table_horizon,
                    'xi_table_horizon': xi_table_horizon,
                    'bound_table_horizon': bound_table_horizon}

        np.savez('data_lq_mpc_multipleSys.npz', **out_dict)
        return out_dict


class Plotter_PF_LQMPC:
    """
    This class is used to plot the performance surface in 3D for different initial points.
    """

    def __init__(self, data_surface, data_xi, data_alpha, data_beta,
                 N_min, N_max, e_power_min, e_power_max):
        """
        This is the initialization of the class
        :param data_surface: A dictionary that contains the data for plotting the surface
        :param data_xi: The xi data with varying error and horizon
        :param data_alpha: The alpha data with varying error and horizon
        :param data_beta: The beta data with varying error and horizon
        :param N_min: minimum used horizon
        :param N_max: maximum used horizon
        :param e_power_min: minimum used power of the error
        :param e_power_max: maximum used power of the error
        """
        self.X = data_surface['X']
        self.Y = data_surface['Y']
        self.J_MPC_true = data_surface['J_MPC_true']
        self.J_MPC_bound = data_surface['J_MPC_bound']
        self.V_OPC = data_surface['V_OPC']
        self.data_xi = data_xi
        self.data_alpha = data_alpha
        self.data_beta = data_beta
        self.horizon = np.arange(N_min, N_max + 1)
        self.error = np.array([10 ** i for i in range(e_power_min, e_power_max + 1)])

    def plot_plane_comparison(self, size, font_type, font_size, color_dict):
        """
        This function plots all the three performance surface.
        :return: NONE
        """
        # Plot
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf_mpc_true = ax.plot_surface(self.X, self.Y, self.J_MPC_true, color=color_dict['C0'],
                                        alpha=0.2, edgecolor='black')
        surf_mpc_bound = ax.plot_surface(self.X, self.Y, self.J_MPC_bound, color=color_dict['C1'],
                                         alpha=0.2, edgecolor='black')
        surf_opc = ax.plot_surface(self.X, self.Y, self.V_OPC, color=color_dict['C2'],
                                   alpha=0.2, edgecolor='black')

        # Calculate the color values for the data points based on Z values
        color_mpc_true = gradient_color(self.J_MPC_true, color_dict['C0'])
        color_mpc_bound = gradient_color(self.J_MPC_bound, color_dict['C1'])
        color_surf_opc = gradient_color(self.V_OPC, color_dict['C2'])

        # Plot scatter markers for each data point with color based on Z values
        sc_true = ax.scatter(self.X.flatten(), self.Y.flatten(), self.J_MPC_true.flatten(), c=color_mpc_true, s=50)
        sc_bound = ax.scatter(self.X.flatten(), self.Y.flatten(), self.J_MPC_bound.flatten(), c=color_mpc_bound, s=50)
        sc_opc = ax.scatter(self.X.flatten(), self.Y.flatten(), self.V_OPC.flatten(), c=color_surf_opc, s=50)

        ax.legend([surf_mpc_true, surf_mpc_bound, surf_opc],
                  ['$J^{[\hat{\mu}_N]}_{\infty}$', '$J_{\mathrm{bound}}$', '$V_{\infty}$'], loc='upper left',
                  fontsize=font_size["legend"], prop={'family': font_type, 'size': font_size["legend"]})
        # ax.legend([sc_Z, sc_R], ['Z = X * Y', 'R = X^2 + Y^2'], loc='upper right')

        # Labels
        ax.set_xlabel(r'$x_1$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        ax.set_ylabel(r'$x_2$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        ax.set_zlabel('Cost Value')
        ax.set_zscale('log')
        plt.savefig('performance_comparison.svg', format='svg', dpi=300)

        plt.show()

    def plot_fc_ec(self, size, font_type, font_size, color_dict, error_level, horizon_nominal):
        """
        This function plots the behavior of the error_consistent functions
        :param size: figure size
        :param font_type: font type
        :param font_size: font sizes of the titles, labels and legends
        :param color_dict: dictionary with colors
        :param error_level: the error_level of the nominal chosen error
        :param horizon_nominal: the nominal prediction horizon
        :return: NONE
        """
        fig, axs = plt.subplots(3, 2, figsize=(size[0] * 2, size[1] * 3))
        # ----------------- Plots about alpha -------------------
        # alpha changes w.r.t. error
        axs[0, 0].plot(self.error, self.data_alpha['error'],
                       label=r'$\alpha_N$',
                       marker='o', markersize=8, markerfacecolor=color_dict['C3'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C0'])
        axs[0, 0].set_title(r'$N = {}$'.format(horizon_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[0, 0].set_xlabel(r'$\delta$', fontdict={'family': font_type, 'size': font_size["label"],
                                                    'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[0, 0].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[0, 0].set_xscale('log')
        axs[0, 0].grid(True, linestyle='--', color='gray', linewidth=0.5)
        # alpha changes w.r.t. horizon
        axs[0, 1].plot(self.horizon, self.data_alpha['horizon'],
                       label=r'$\alpha_N$',
                       marker='s', markersize=8, markerfacecolor=color_dict['C4'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C0'])
        axs[0, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_level),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[0, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[0, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        # axs[0, 1].set_yscale('log')
        axs[0, 1].set_xticks(self.horizon)
        axs[0, 1].grid(True, linestyle='--', color='gray', linewidth=0.5)
        # axs[0, 1].set_xticklabels(self.horizon)

        # ----------------- Plots about beta -------------------
        # beta changes w.r.t. error
        axs[1, 0].plot(self.error, self.data_beta['error'],
                       label=r'$\beta_N$',
                       marker='o', markersize=8, markerfacecolor=color_dict['C3'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C1'])
        axs[1, 0].set_title(r'$N = {}$'.format(horizon_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[1, 0].set_xlabel(r'$\delta$', fontdict={'family': font_type, 'size': font_size["label"],
                                                    'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[1, 0].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[1, 0].set_xscale('log')
        axs[1, 0].grid(True, linestyle='--', color='gray', linewidth=0.5)
        # beta changes w.r.t. horizon
        axs[1, 1].plot(self.horizon, self.data_beta['horizon'],
                       label=r'$\beta_N$',
                       marker='s', markersize=8, markerfacecolor=color_dict['C4'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C1'])
        axs[1, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_level),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[1, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[1, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        # axs[1, 1].set_yscale('log')
        axs[1, 1].set_xticks(self.horizon)
        axs[1, 1].grid(True, linestyle='--', color='gray', linewidth=0.5)

        # ----------------- Plots about xi -------------------
        # xi changes w.r.t. error
        axs[2, 0].plot(self.error, self.data_xi['error'],
                       label=r'$\xi_N$',
                       marker='o', markersize=8, markerfacecolor=color_dict['C3'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C2'])
        axs[2, 0].set_title(r'$N = {}$'.format(horizon_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[2, 0].set_xlabel(r'$\delta$', fontdict={'family': font_type, 'size': font_size["label"],
                                                    'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[2, 0].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[2, 0].set_xscale('log')
        axs[2, 0].grid(True, linestyle='--', color='gray', linewidth=0.5)
        # xi changes w.r.t. horizon
        axs[2, 1].plot(self.horizon, self.data_xi['horizon'],
                       label=r'$\xi_N$',
                       marker='s', markersize=8, markerfacecolor=color_dict['C4'],
                       markeredgewidth=2, markeredgecolor='blue',
                       linewidth=2, color=color_dict['C2'])
        axs[2, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_level),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[2, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[2, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        # axs[2, 1].set_yscale('log')
        axs[2, 1].set_xticks(self.horizon)
        axs[2, 1].grid(True, linestyle='--', color='gray', linewidth=0.5)

        # Adjust layout to prevent overlapping of subplots
        plt.tight_layout()
        plt.savefig('error_consistent_curves.svg', format='svg', dpi=300)

        # Show the plot
        plt.show()


class Plotter_PF_LQMPC_Multiple:
    """
    This class is used for plotting the performance for multiple systems
    """

    def __init__(self, color_dict: dict, fig_size: np.ndarray,
                 font_type: str, font_size: dict):
        """
        This is the initialization of the plotter
        :param color_dict: The color dictionary
        :param fig_size: The size of the figure
        :param font_type: the type of the font
        :param font_size: a dictionary containing the size of the font
        """
        self.colors = color_dict
        self.fig_size = fig_size
        self.font_type = font_type
        self.font_size = font_size

    def plotter_error(self, N_nominal: int, error: np.ndarray,
                      alpha_table: np.ndarray, beta_table: np.ndarray,
                      xi_table: np.ndarray, bound_table: np.ndarray) -> None:
        """
        This function plot the curves for different error level
        :param N_nominal: the nominal prediction horizon
        :param error: the error vector
        :param alpha_table: the alpha table
        :param beta_table: the beta table
        :param xi_table: the xi table
        :param bound_table: the bound
        :return: None (Simply do the plots, no return)
        """
        # initialize the figure
        fig, ax = plt.subplots(2, 2,
                               figsize=(self.fig_size[0] * 2, self.fig_size[1] * 2))

        # specify the text
        alpha_text = {'title': r'$N = {}$'.format(N_nominal),
                      'x_label': r'$\delta$',
                      'data': r'$\alpha_N$'}
        beta_text = {'title': r'$N = {}$'.format(N_nominal),
                     'x_label': '$\delta$',
                     'data': r'$\beta_N$'}
        xi_text = {'title': r'$N = {}$'.format(N_nominal),
                   'x_label': r'$\delta$',
                   'data': r'$\xi_N$'}
        bound_text = {'title': r'$N = {}$'.format(N_nominal),
                      'x_label': r'$\delta$',
                      'data': r'$J_{\mathrm{bound}}$'}

        # do the plotting
        statistical_continuous(ax[0, 0], error, alpha_table,
                               alpha_text, self.colors['C0'],
                               self.font_type, self.font_size)
        statistical_continuous(ax[0, 1], error, beta_table,
                               beta_text, self.colors['C1'],
                               self.font_type, self.font_size)
        statistical_continuous(ax[1, 0], error, xi_table,
                               xi_text, self.colors['C2'],
                               self.font_type, self.font_size)
        statistical_continuous(ax[1, 1], error, bound_table,
                               bound_text, self.colors['C3'],
                               self.font_type, self.font_size,
                               y_scale_log=True)
        # Show and save
        plt.tight_layout()
        plt.savefig('error_variation.svg', format='svg', dpi=500)
        plt.show()

    def plotter_horizon(self, err_nominal: float, horizon: np.ndarray,
                        alpha_table: np.ndarray, beta_table: np.ndarray,
                        xi_table: np.ndarray, bound_table: np.ndarray) -> None:
        """
        This function plot the curves for different error level
        :param err_nominal: the nominal modeling error
        :param horizon: the horizon vector
        :param alpha_table: the alpha table
        :param beta_table: the beta table
        :param xi_table: the xi table
        :param bound_table: the bound
        :return: None (Simply do the plots, no return)
        """
        # initialize the figure
        fig, ax = plt.subplots(2, 2,
                               figsize=(self.fig_size[0] * 2, self.fig_size[1] * 2))

        # specify the text
        alpha_text = {'title': r'$\delta_A = \delta_B = {}$'.format(err_nominal),
                      'x_label': r'$N$',
                      'data': r'$\alpha_N$'}
        beta_text = {'title': r'$\delta_A = \delta_B = {}$'.format(err_nominal),
                     'x_label': r'$N$',
                     'data': r'$\beta_N$'}
        xi_text = {'title': r'$\delta_A = \delta_B = {}$'.format(err_nominal),
                   'x_label': r'$N$',
                   'data': r'$\xi_N$'}
        bound_text = {'title': r'$\delta_A = \delta_B = {}$'.format(err_nominal),
                      'x_label': r'$N$',
                      'data': r'$J_{\mathrm{bound}}$'}

        # do the plotting
        statistical_continuous(ax[0, 0], horizon, alpha_table,
                               alpha_text, self.colors['C0'],
                               self.font_type, self.font_size,
                               set_x_ticks=True)
        statistical_continuous(ax[0, 1], horizon, beta_table,
                               beta_text, self.colors['C1'],
                               self.font_type, self.font_size,
                               set_x_ticks=True)
        statistical_continuous(ax[1, 0], horizon, xi_table,
                               xi_text, self.colors['C2'],
                               self.font_type, self.font_size,
                               set_x_ticks=True)
        statistical_continuous(ax[1, 1], horizon, bound_table,
                               bound_text, self.colors['C3'],
                               self.font_type, self.font_size,
                               y_scale_log=True, set_x_ticks=True)
        # Show and save
        plt.tight_layout()
        plt.savefig('horizon_variation.svg', format='svg', dpi=500)
        plt.show()

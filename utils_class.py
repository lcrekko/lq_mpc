import math
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from utils import (bar_u_solve, bar_d_u_solve,
                   local_radius, ex_stability_lq, ex_stability_bounds,
                   fc_omega_eta, fc_ec_h, fc_ec_E, fc_omega_eta_extension,
                   circle_generator, gradient_color)

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
            alpha_error[i] = self.calculator.energy_decreasing(N_nominal, self.error_vec[i],
                                                               self.error_vec[i], x, p)['alpha']
            beta_error[i] = self.calculator.energy_decreasing(N_nominal, self.error_vec[i],
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

    def data_generation_plane(self, N, sim_info, sys_true, err_nominal, info_ref, M_V, p, N_points, ratio_ext_radius):
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
                temp_info_ol_mpc = ol_mpc.solve(X_points[:, i], x_ref, u_ref)
                temp_V_OPC[i] = temp_info_ol_mpc['V_N']

                # Computing the cost bound for each of the points
                temp_J_MPC_bound[i] = decreasing_factor * (temp_alpha[i] * temp_V_OPC[i] + temp_beta[i])

                # Computing the closed-loop MPC cost for each of the points, knowing the estimated system
                temp_info_cl_mpc = cl_mpc.simulate(X_points[:, i], A_true, B_true, x_ref, u_ref)
                temp_J_MPC_true[i] = temp_info_cl_mpc['J_T']

            X_1 = np.append(X_1, X_points[1, :])
            X_2 = np.append(X_2, X_points[2, :])
            J_MPC_true = np.append(J_MPC_true, temp_J_MPC_true)
            J_MPC_bound = np.append(J_MPC_true, temp_J_MPC_bound)
            V_OPC = np.append(J_MPC_true, temp_V_OPC)

        X = np.vstack((X_1, X_2))

        return {'X': X, 'J_MPC_true': J_MPC_true, 'J_MPC_bound': J_MPC_bound, 'V_OPC': V_OPC}


class Plotter_PF_LQMPC:
    """
    This class is used to plot the performance surface in 3D for different initial points.
    """

    def __init__(self, data_surface, data_xi, data_alpha, data_beta,
                 N_min, N_max, e_pow_min, e_pow_max):
        """
        This is the initialization of the class
        :param data_surface: A dictionary that contains the data for plotting the surface
        :param data_xi: The xi data with varying error and horizon
        :param data_alpha: The alpha data with varying error and horizon
        :param data_beta: The beta data with varying error and horizon
        :param N_min: minimum used horizon
        :param N_max: maximum used horizon
        :param e_pow_min: minimum used power of the error
        :param e_pow_max: maximum used power of the error
        """
        self.X = data_surface['X']
        self.J_MPC_true = data_surface['J_MPC_true']
        self.J_MPC_bound = data_surface['J_MPC_bound']
        self.V_OPC = data_surface['V_OPC']
        self.data_xi = data_xi
        self.data_alpha = data_alpha
        self.data_beta = data_beta
        self.horizon = np.arange(N_min, N_max + 1)
        self.error = np.array([10 ** i for i in range(e_pow_min, e_pow_max + 1)])

    def plot_plane_comparison(self, size, font_type, font_size, color_dict):
        """
        This function plots all the three performance surface.
        :return: NONE
        """
        # Define the data
        x_1 = self.X[1, :]
        x_2 = self.X[2, :]

        # Plot
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf_mpc_true = ax.plot_surface(x_1, x_2, self.J_MPC_true, color="red", alpha=0.2, edgecolor='black')
        surf_mpc_bound = ax.plot_surface(x_1, x_2, self.J_MPC_bound, color="green", alpha=0.2, edgecolor='black')
        surf_opc = ax.plot_surface(x_1, x_2, self.V_OPC, color="blue", alpha=0.2, edgecolor='black')

        # Calculate the color values for the data points based on Z values
        color_mpc_true = gradient_color(self.J_MPC_true, color_dict['C0'])
        color_mpc_bound = gradient_color(self.J_MPC_bound, color_dict['C1'])
        color_surf_opc = gradient_color(self.V_OPC, color_dict['C2'])

        # Plot scatter markers for each data point with color based on Z values
        ax.scatter(x_1, x_2, self.J_MPC_true, c=color_mpc_true, s=50)
        ax.scatter(x_1, x_2, self.J_MPC_bound, c=color_mpc_bound, s=50)
        ax.scatter(x_1, x_2, self.V_OPC, c=color_surf_opc, s=50)

        ax.legend([surf_mpc_true, surf_mpc_bound, surf_opc],
                  ['$J^{[\hat{\mu}_N]}_{\infty}$', '$J_{\text{bound}}$', '$V_{\infty}$'], loc='upper left',
                  fontsize=font_size["legend"], prop={'family': font_type, 'size': font_size["legend"]})
        # ax.legend([sc_Z, sc_R], ['Z = X * Y', 'R = X^2 + Y^2'], loc='upper right')

        # Labels
        ax.set_xlabel(r'$x_1$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        ax.set_ylabel(r'$x_2$', fontdict={'family': font_type, 'size': font_size["label"], 'weight': 'bold'})
        ax.set_zlabel('Cost Value')
        ax.set_zscale('log')

        plt.show()

    def plot_fc_ec(self, size, font_type, font_size, error_nominal, horizon_nominal):
        """
        This function plots the behavior of the error_consistent functions
        :param size: figure size
        :param font_type: font type
        :param font_size: font sizes of the titles, labels and legends
        :param error_nominal: the nominal error
        :param horizon_nominal: the nominal prediction horizon
        :return: NONE
        """
        fig, axs = plt.subplots(3, 2, figsize=(size[0] * 2, size[1] * 3))
        # ----------------- Plots about alpha -------------------
        # alpha changes w.r.t. error
        axs[0, 0].plot(self.error, self.data_alpha['error'],
                       label=r'$\alpha_N$')
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
        # alpha changes w.r.t. horizon
        axs[0, 1].plot(self.horizon, self.data_alpha['horizon'],
                       label=r'$\alpha_N$')
        axs[0, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[0, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[0, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[0, 1].set_yscale('log')

        # ----------------- Plots about beta -------------------
        # beta changes w.r.t. error
        axs[1, 0].plot(self.error, self.data_beta['error'],
                       label=r'\beta_N')
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
        # beta changes w.r.t. horizon
        axs[1, 1].plot(self.horizon, self.data_beta['horizon'],
                       label=r'$\beta_N$')
        axs[1, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[1, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[1, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[1, 1].set_yscale('log')

        # ----------------- Plots about xi -------------------
        # xi changes w.r.t. error
        axs[2, 0].plot(self.error, self.data_xi['error'],
                       label=r'\xi_N')
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
        # xi changes w.r.t. horizon
        axs[2, 1].plot(self.horizon, self.data_xi[2, :],
                       label=r'$\xi_N$')
        axs[2, 1].set_title(r'$\delta_A = \delta_B = {}$'.format(error_nominal),
                            fontdict={'family': font_type, 'size': font_size["label"],
                                      'weight': 'bold'})
        axs[2, 1].set_xlabel(r'$N$', fontdict={'family': font_type, 'size': font_size["label"],
                                               'weight': 'bold'})
        # axs[0, 0].set_ylabel('Y Label')
        axs[2, 1].legend(loc='upper left',
                         fontsize=font_size["legend"],
                         prop={'family': font_type, 'size': font_size["legend"]})
        axs[2, 1].set_yscale('log')

        # Adjust layout to prevent overlapping of subplots
        plt.tight_layout()

        # Show the plot
        plt.show()

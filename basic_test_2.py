import numpy as np
import control as ct
from utils import ex_stability_lq, my_eigen, N_incremental_test
# import math
import matplotlib.pyplot as plt

# Specify the estimated system -- We use marginal stable
A = np.array([[1, 0.7], [0.12, 0.4]])
B = np.array([[1], [1.2]])

print(f"The spectral radius of matrix A is", {np.max(np.linalg.eigvals(A))})

# Specify the cost -- We use the identity as the most simple case
Q = 2 * np.eye(2)
R = np.eye(1)

infoQ = my_eigen(Q)
infoR = my_eigen(R)

print(infoQ)
print(infoR)

# compute the LQR gain
K_lqr, P_lqr, eig_lqr = ct.dlqr(A, B, Q, R)
norm_K_lqr = np.linalg.norm(K_lqr, ord=2)

A_cl = A - B * K_lqr

my_rho = np.max(np.abs(np.linalg.eigvals(A_cl)))

big_K = 100

norm_K = np.zeros(big_K)
est = np.zeros(big_K)

myM = np.eye(2)
# myM_2 = np.eye(2)

for i in range(big_K):
    norm_K[i] = np.linalg.norm(myM, ord=2)
    est[i] = 1.1 * ((my_rho + 0.4) ** i)
    myM = myM @ A_cl
    # myM_2 = myM.T @ myM

plt.figure()
plt.plot(norm_K)
plt.plot(est)
plt.legend(['norm', 'est'])
plt.show()

info_lqr_ex = ex_stability_lq(A, B, Q, R, -K_lqr)
N_plus = N_incremental_test(A, Q, info_lqr_ex['gamma'], info_lqr_ex['rho_gamma'])
print(f"The norm of the lqr gain is", {norm_K_lqr})
print(f"The coefficient C_K", {info_lqr_ex['C_K']})
print(f"The coefficient lambda_K", {info_lqr_ex['lambda_K']})
print(f"The coefficient rho_K", {info_lqr_ex['rho_K']})
print(f"The coefficient gamma", {info_lqr_ex['gamma']})
print(f"The coefficient rho_gamma", {info_lqr_ex['rho_gamma']})
print('\n')
print(f"Extra prediction horizon", {N_plus})

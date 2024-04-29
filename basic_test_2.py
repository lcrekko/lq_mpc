import numpy as np
import control as ct
import matplotlib.pyplot as plt

# Specify the estimated system -- We use marginal stable
A = np.array([[1, 0.15], [0.1, 1]])
B = np.array([[0.1], [1.1]])

# Specify the cost -- We use the identity as the most simple case
Q = np.eye(2)
R = np.eye(1)

K_lqr, P_lqr, eig_lqr = ct.dlqr(A, B, Q, R)

print(eig_lqr)

A_cl = A - B * K_lqr

big_K = 100

norm_K = np.zeros(big_K)
est = np.zeros(big_K)

myM = np.eye(2)

for i in range(big_K):
    norm_K[i] = np.linalg.norm(myM)
    est[i] = 1.6 * (0.86 ** i)
    myM = myM @ A_cl

plt.figure()
plt.plot(norm_K)
plt.plot(est)
plt.legend(['norm', 'est'])
plt.show()

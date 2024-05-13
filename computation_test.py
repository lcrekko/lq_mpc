"""
This script is used for generating error matrix, the generated matrix is saved
as npy files, we did for both Frobenius norm and the 2 norm, and we set the number of
generated matrix to be 25

The time used for generating the matrix is rather short (approximately 5 seconds)
"""
from utils import generate_random_matrix, random_matrix, error_matrix_generator
import numpy as np

A = np.zeros([2, 2])
B = np.zeros([2, 1])
e_info = {'e_min': 10 ** (-3), 'e_max': 10 ** (-2)}
error_vec = np.linspace(e_info['e_min'], e_info['e_max'], 10)

N_matrix = 5

output = error_matrix_generator(A, B, error_vec, N_matrix, '2')

'''
print(output['error_A'].shape)
print(np.linalg.norm(output['error_A'][:, :, 1, 0], ord=2))
print(np.linalg.norm(output['error_A'][:, :, 6, 0], ord=2))
'''

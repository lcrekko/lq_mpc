import math

import numpy as np
from utils_class import Plotter_MPC

A = np.random.rand(3, 30)
B = np.random.rand(2, 30)

fig_height = 4
fig_size = np.array([fig_height * 0.5 * (math.sqrt(5) + 1), fig_height])

my_font_type = "Times New Roman"
my_font_size = {"title": fig_height * 4, "label": fig_height * 3, "legend": fig_height * 3}

my_plot = Plotter_MPC(A, B)
my_plot.plot_1_D_state(fig_size, my_font_type, my_font_size)

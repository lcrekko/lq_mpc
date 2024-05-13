import numpy as np
import matplotlib.pyplot as plt
import math
from utils import statistical_continuous

# we are not going to test a plot that shows the mean, variance, the maximum and minimum value
n_points = 20
n_value = 20

x_data = np.linspace(0, 1, n_points)
y_data_1 = np.random.rand(n_value, n_points)
y_data_2 = 2 * np.random.rand(n_value, n_points)

color_1 = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
color_2 = (1.0, 0.4980392156862745, 0.054901960784313725)

fig_height_rectangle = 4
fig_height_square = 3
font_type = "Times New Roman"
fig_size_rectangle = np.array([fig_height_rectangle * 0.5 * (math.sqrt(5) + 1),
                               fig_height_rectangle])
font_size_rectangle = {"title": fig_height_square * 8, "label": fig_height_square * 8,
                       "legend": fig_height_square * 8}

info_text_1 = {'title': r"$\xi$", 'x_label': "$x$", 'data': '$y_1$'}
info_text_2 = {'title': "data 2", 'x_label': "$x$", 'data': '$y_2$'}

fig, ax = plt.subplots(2, 1,
                       figsize=(fig_size_rectangle[0] * 1, fig_size_rectangle[1] * 2))

statistical_continuous(ax[0], x_data, y_data_1,
                       info_text_1, color_1, font_type, font_size_rectangle)
statistical_continuous(ax[1], x_data, y_data_2,
                       info_text_2, color_2, font_type, font_size_rectangle)

plt.tight_layout()
plt.show()
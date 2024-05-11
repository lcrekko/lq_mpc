### TO DO LIST

##### 2020-05-10

1. Recoding the matrix generator to make it contains the following functionalities:
    - make sure it includes extreme cases touching the boundary norm by **explicitly specifying them**
    - make sure the generated matrices are close to the extreme case by **setting the lower bar**
2. Find a way to save the generated matrices in the current project and *reuse* them in running the code
    - maybe try ask Chatgpt what is the similar function in python to the `import` in MATLAB
3. Recompute the error-consistent function curves such that they are in the form of **line plot with confidence interval
   **
    - First Create another class maybe for generate relevant data
    - modify the existing `plot_fc_ec` function in the `Plotter_PF_LQMPC` class to make it plotting the shaded area
      using `fill_between`
    - use the command `plt.figure(facecolor='lightgrey')` to change the background to blue
    - use the command `plt.grid(color='white')` to change the grid color to white
4. Create a separate figure to show how the performance bound and true performance is as varying the modeling error and
   the horizon
5. The 3D plot can be abandoned
6. For the variation in horizon, it is recommended to use violin plot
7. The final figure should be 2 separate ones, each of which is of size 2-by-2

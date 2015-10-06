# Python code for mlai2015 lectures.

# import the time model to allow python to pause.
import time
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML

def hyperplane_coordinates(w, b, plot_limits):
    """Helper function for plotting the decision boundary of the perceptron."""
    if abs(w[1])>abs(w[0]):
        # If w[1]>w[0] in absolute value, plane is likely to be leaving tops of plot.
        x0 = plot_limits['x']
        x1 = -(b + x0*w[0])/w[1]
    else:
        # otherwise plane is likely to be leaving sides of plot.
        x1 = plot_limits['y']
        x0 = -(b + x1*w[1])/w[0]
    return x0, x1

def init_perceptron_plot(f, ax, x_plus, x_minus, w, b, x_select):
    """Initialise a plot for showing the perceptron decision boundary."""

    h = {}

    ax[0].set_aspect('equal')
    # Plot the data again
    ax[0].plot(x_plus[:, 0], x_plus[:, 1], 'rx')
    ax[0].plot(x_minus[:, 0], x_minus[:, 1], 'go')
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)
    strt = -b/w[1]

    h['arrow'] = ax[0].arrow(0, strt, w[0], w[1]+strt, head_width=0.2)
    # plot a line to represent the separating 'hyperplane'
    h['plane'], = ax[0].plot(x0, x1, 'b-')
    ax[0].set_xlim(plot_limits['x'])
    ax[0].set_ylim(plot_limits['y'])
    ax[0].set_xlabel('$x_0$', fontsize=20)
    ax[0].set_ylabel('$x_1$', fontsize=20)
    h['iter'] = ax[0].set_title('Update 0')

    h['select'], = ax[0].plot(x_select[0], x_select[1], 'ro', markersize=10)

    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')
    return h

def update_perceptron_plot(h, f, ax, x_plus, x_minus, i, w, b, x_select):
    """Update plots after decision boundary has changed."""
    # Helper function for updating plots
    h['select'].set_xdata(x_select[0])
    h['select'].set_ydata(x_select[1])
    # Re-plot the hyper plane 
    plot_limits = {}
    plot_limits['x'] = np.asarray(ax[0].get_xlim())
    plot_limits['y'] = np.asarray(ax[0].get_ylim())
    x0, x1 = hyperplane_coordinates(w, b, plot_limits)
    strt = -b/w[1]
    h['arrow'].remove()
    del(h['arrow'])
    h['arrow'] = ax[0].arrow(0, strt, w[0], w[1]+strt, head_width=0.2)
    
    h['plane'].set_xdata(x0)
    h['plane'].set_ydata(x1)

    h['iter'].set_text('Update ' + str(i))
    ax[1].cla()
    bins = 15
    f_minus = np.dot(x_minus, w)
    f_plus = np.dot(x_plus, w)
    ax[1].hist(f_plus, bins, alpha=0.5, label='+1', color='r')
    ax[1].hist(f_minus, bins, alpha=0.5, label='-1', color='g')
    ax[1].legend(loc='upper right')

    display(f)
    clear_output(wait=True)
    if i<3:
        time.sleep(0.5)
    else:
        time.sleep(.25)   
    return h

def init_regression_plot(f, ax, x, y, m_vals, c_vals, E_grid, m_star, c_star):
    """Function to plot the initial regression fit and the error surface."""
    h = {}
    levels=[0, 0.5, 1, 2, 4, 8, 16, 32, 64]
    h['cont'] = ax[0].contour(m_vals, c_vals, E_grid, levels=levels) # this makes the contour plot on axes 0.
    plt.clabel(h['cont'], inline=1, fontsize=15)
    ax[0].set_xlabel('$m$', fontsize=20)
    ax[0].set_ylabel('$c$', fontsize=20)
    h['msg'] = ax[0].set_title('Error Function', fontsize=20)

    # Set up plot
    h['data'], = ax[1].plot(x, y, 'r.', markersize=10)
    ax[1].set_xlabel('$x$', fontsize=20)
    ax[1].set_ylabel('$y$', fontsize=20)
    ax[1].set_ylim((-9, -1)) # set the y limits of the plot fixed
    ax[1].set_title('Best Fit', fontsize=20)

    # Plot the current estimate of the best fit line
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plotting the current best line fit.
    y_plot = m_star*x_plot + c_star
    h['fit'], = ax[1].plot(x_plot, y_plot, 'b-', linewidth=3)
    return h

def update_regression_plot(h, f, ax, m_star, c_star, iteration):
    """Update the regression plot with the latest fit and position in error space."""
    ax[0].plot(m_star, c_star, 'g*')
    x_plot = np.asarray(ax[1].get_xlim()) # get the x limits of the plot for plo
    y_plot = m_star*x_plot + c_star
    
    # show the current status on the plot of the data
    h['fit'].set_ydata(y_plot)
    h['msg'].set_text('Iteration '+str(iteration))
    display(f)
    clear_output(wait=True)
    time.sleep(0.25) # pause between iterations to see update
    return h

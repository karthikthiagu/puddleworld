import os
import numpy as np

from matplotlib import pyplot as plt

def configure():

    # Set the plot
    paperwidth = 11.7
    paperheight = 8.3
    
    margin = 1.0
    plt.figure(1, figsize = (paperwidth - 2 * margin, paperheight - 2*margin))
    plt.margins(0.1, 0.0)
    plt.xticks([1] + list(np.arange(0, 5001, 1000))[1:])
    plt.yticks(np.arange(0, 11, 2))
    plt.ylim((0, 12))
    
    margin = 0.8
    plt.figure(2, figsize = (paperwidth - 2 * margin, paperheight - 2*margin))
    plt.margins(0.1, 0.0)
    plt.xticks([1] + list(np.arange(0, 5001, 1000))[1:])
    plt.yticks(np.arange(0, 100, 10))
    plt.ylim((0, 101))

def plot(x, y, simulation):
    plt.figure(simulation['figure'])
    # Plotting
    plt.plot(x, y, color = simulation['color'])
    # Labeling
    plt.xlabel(simulation['xlabel'])
    plt.ylabel(simulation['ylabel'], rotation = 'horizontal', labelpad = 30)

    if 'filename' in simulation:
        plt.savefig(simulation['filename'])
        print 'saving plots'   

def compute():
    experiments = 50
    returns = np.zeros(5000)
    steps = np.zeros(5000)
    for index in range(experiments):
        temp_returns = np.load( open('results/qlearning/%d_returns.npy' % (index + 1), 'r' ) )
        temp_steps   = np.load( open('results/qlearning/%d_steps.npy' % (index + 1)  , 'r' ) )
        returns += (1 / float(index + 1) ) * (temp_returns - returns)   # Stochastic averaging rule
        steps   += (1 / float(index + 1) ) * (temp_steps - steps)       # Stochastic averaging rule
    return (returns, steps)
    
if __name__ == '__main__':
    configure()
    returns, steps = compute()
    episodes = range(5001)[1 : ]
    returns_plot = {'figure' : 1, 'color' : 'blue', 'xlabel' : 'Episodes', 'ylabel' : 'Average\n reward', 'filename' : 'plots/qlearning/returns.png'}
    steps_plot   = {'figure' : 2, 'color' : 'blue', 'xlabel' : 'Episodes', 'ylabel' : 'Average\n episode\n length', 'filename' : 'plots/qlearning/steps.png'}
    plot(episodes, returns, returns_plot)
    plot(episodes, steps, steps_plot)


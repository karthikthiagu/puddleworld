import os
import sys
import numpy as np
from plotter import plot, configure

def compute(experiments, episodes, sarsa_lambda, goal):
    returns = np.zeros(episodes / 10)
    steps = np.zeros(episodes / 10)
    for index in range(experiments):
        temp_returns = np.load( open('results/sarsa_lambda/%d-%s-%s_returns.npy' % (index + 1, goal, sarsa_lambda), 'r' ) )
        temp_steps   = np.load( open('results/sarsa_lambda/%d-%s-%s_steps.npy' % (index + 1, goal, sarsa_lambda)  , 'r' ) )
        returns += (1 / float(index + 1) ) * (temp_returns - returns)   # Stochastic averaging rule
        steps   += (1 / float(index + 1) ) * (temp_steps - steps)       # Stochastic averaging rule
    return (returns, steps)

def conclude(xepisodes, returns, steps, goal):
    configure()
    returns_plot = {'figure' : 1, 'color' : 'blue', 'xlabel' : 'Episodes', 'ylabel' : 'Average\n reward',\
                    'filename' : 'plots/sarsa_lambda/%s-returns.png' % goal}
    steps_plot   = {'figure' : 2, 'color' : 'blue', 'xlabel' : 'Episodes', 'ylabel' : 'Average\n episode\n length',\
                    'filename' : 'plots/sarsa_lambda/%s-steps.png' % goal}
    plot(xepisodes, returns, returns_plot)
    plot(xepisodes, steps, steps_plot)

if __name__ == '__main__':
    os.system('clear')

    if len(sys.argv) < 4:
        print 'Usage : python scripts/sarsa_lambda/plotExperiment.py experiments episodes per_episode sarsa_lambda goal'
        exit()
    else:
        _, experiments, episodes, per_episode, sarsa_lambda, goal = sys.argv
        experiments = int(experiments)
        episodes = int(episodes)
        per_episode = int(per_episode)

    xepisodes = range(1, episodes, per_episode)                             # X-axis of the goal
    returns, steps = compute(experiments, episodes, sarsa_lambda, goal)                   # Compute the average returns and steps from all experiments
    conclude(xepisodes, returns, steps, goal)                               # Conclude the experiments by plotting the returns and steps


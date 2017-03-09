import os
import sys

import numpy as np

import rlglue.RLGlue as RLGlue


# A class to run a single experiment
class Experiment:
    
    def __init__(self, episodes, identifier, goal):
        RLGlue.RL_init()    # Initialize the RL-agent
        self.goal = goal    # The problem being solved, aka the goal state
        self.setGoalState() # This function sets the goal state for the given problem
        self.identifier = '%s-%s' % (str(identifier), str(self.goal) )  # The identifier is unique to this experiment, all data will be saved with this identifier
        self.episodes = episodes    # Number of episodes that this experiment will be run for
        self.test_per = 10
        self.returns = np.zeros( (self.episodes / self.test_per, ) )    # A container to store the returns per episode, indexed by the episode
        self.steps = np.zeros( (self.episodes / self.test_per, ) )      # A container to store the steps-to-goal per episode, indexed by the episode
        print '---------------------'
        print 'Initializing agent %s' % self.identifier

    # Set the goal state depending on the problem being solved
    def setGoalState(self):
        if self.goal == 'A':
            RLGlue.RL_env_message("set-goal-state 1 12")
        elif self.goal == 'B':
            RLGlue.RL_env_message("set-goal-state 3 10")
        elif self.goal == 'C':
            RLGlue.RL_env_message("set-goal-state 7 8")
            RLGlue.RL_env_message("turn-off-wind")

    def save(self):
        RLGlue.RL_agent_message("save_policy results/sarsa/%s.dat" % self.identifier)   # Save the value function
        returns = open('results/sarsa/%s_returns.npy' % self.identifier, 'w') # File to save the per-episode returns
        steps   = open('results/sarsa/%s_steps.npy'   % self.identifier, 'w')   # File to save the per-episode steps-to-the-goal
        np.save(returns, self.returns)
        np.save(steps, self.steps)
        RLGlue.RL_cleanup()     # Cleanup the agent's memory so that another agent can be instantiated
        print 'Saving and cleaning up agent %s' % self.identifier
        print '-------------------------------'

    def train(self):
        for episode in range(self.episodes):
            if (episode + 1) % 500 == 0:
                print '%d episodes done.' % (episode + 1)
            RLGlue.RL_episode(1000)     # Run each episode, cap the num_steps to 1000 => episode will terminate at 1000
            if episode % self.test_per == 0:
                self.test(episode / self.test_per)          # Test for every 20 episodes

    def test(self, episode):
        RLGlue.RL_agent_message("freeze learning")
        test_episodes = 5
        for test_episode in range(test_episodes):
            RLGlue.RL_episode(1000)
            self.returns[episode] += RLGlue.RL_return()
            self.steps[episode]   += RLGlue.RL_num_steps()
        self.returns[episode] /= float(test_episodes)        # Sum of rewards in the episode; returns is actually a misnomer
        self.steps[episode] /= float(test_episodes)          # Nuber of steps to goal in the episode
        RLGlue.RL_agent_message("unfreeze learning")


if __name__ == '__main__':
    os.system('clear')

    if len(sys.argv) < 3:
        print 'Usage : python scripts/sarsa/experiment.py experiment_start experiment_end episodes goal'
        exit()
    else:
        _, experiment_start, experiment_end, episodes, goal = sys.argv
        experiment_start = int(experiment_start)
        experiment_end = int(experiment_end)
        episodes = int(episodes)

    for index in range(experiment_start, experiment_end + 1, 1):
        experiment = Experiment(episodes, '%d' % index, goal)               # Initialize the agent
        experiment.train()                                                  # Train the agent
        experiment.save()                                                   # Save agent details
    

import os
import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
import numpy as np

class SarsaAgent(Agent):
    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()
    sarsa_stepsize = 0.5
    sarsa_epsilon = 0.1
    sarsa_gamma = 0.9
    numStates = 0
    numActions = 0
    value_function = None

    episode_count = 0
    num_steps_in_episode = 0
    episode_return = 0

    policyFrozen=False
    exploringFrozen=False
	
    def agent_init(self,taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        if TaskSpec.valid:
            assert len(TaskSpec.getIntObservations())==1, "expecting 1-dimensional discrete observations"
            assert len(TaskSpec.getDoubleObservations())==0, "expecting no continuous observations"
            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"
            self.numStates=TaskSpec.getIntObservations()[0][1]+1;

            assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
            assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
            self.numActions=TaskSpec.getIntActions()[0][1]+1;
            
            self.value_function=[self.numActions*[0.0] for i in range(self.numStates)]

        else:
            print "Task Spec could not be parsed: "+taskSpecString;
            
        self.lastAction=Action()
        self.lastObservation=Observation()
        
    def egreedy(self, state):
        if not self.exploringFrozen and self.randGenerator.random()<self.sarsa_epsilon:
            return self.randGenerator.randint(0,self.numActions-1)

        max_indices = np.where( np.array(self.value_function[state]) == np.array(self.value_function[state]).max() )[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return np.random.choice(max_indices)
        
    # Start the episode by taking an action from the start-state
    def agent_start(self,observation):
        self.episode_count += 1
        self.num_steps_in_episode = 1
        self.episode_return = 0
        # Observe the initial state
        theState=observation.intArray[0]
        # Take an action in this state using an epsilon-greedy policy w.r.t the current value function
        thisIntAction=self.egreedy(theState)
        # Store this action
        returnAction=Action()
        returnAction.intArray=[thisIntAction]
        
        # Make this action the last action
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)

        return returnAction

    # From <lastState> the agent took <lastAction> and moved to <observation> getting a <reward> on the way
    # Now update the q-values (action-values) for <lastState, lastAction> by bootstrapping the value of <newState>
    def agent_step(self,reward, observation):
        self.num_steps_in_episode += 1
        self.episode_return += reward

        lastState = self.lastObservation.intArray[0]      # S --- S_t
        lastAction = self.lastAction.intArray[0]          # A --- A_t
        newState = observation.intArray[0]                # S --- S_t+1
        #reward                                           # R --- R_t+1
        newIntAction=self.egreedy(newState)               # A --- A_t+1

        Q_sa = self.value_function[lastState][lastAction] # Q(S_t, A_t)
        Q_sprime_aprime = self.value_function[newState][newIntAction] # Q(S_t+1, A_t+1)

        new_Q_sa = Q_sa + self.sarsa_stepsize * (reward + self.sarsa_gamma * Q_sprime_aprime - Q_sa)

        if not self.policyFrozen:
            self.value_function[lastState][lastAction]=new_Q_sa

        returnAction = Action()   # This is the action that will be returned
        returnAction.intArray = [newIntAction]
        
        self.lastAction = copy.deepcopy(returnAction)         # For the next pass, A_t+1 will be the action
        self.lastObservation = copy.deepcopy(observation)     # For the next pass, S_t+1 will be the state

        return returnAction

    def agent_end(self,reward):
        lastState=self.lastObservation.intArray[0]
        lastAction=self.lastAction.intArray[0]

        Q_sa = self.value_function[lastState][lastAction]

        new_Q_sa = Q_sa + self.sarsa_stepsize * (reward - Q_sa)

        if not self.policyFrozen:
            self.value_function[lastState][lastAction] = new_Q_sa

        self.episode_return += reward
        #print 'episode = %d, steps = %d, reward = %d' % (self.episode_count, self.num_steps_in_episode, self.episode_return)

    def agent_cleanup(self):
        pass

    def save_value_function(self, fileName):
        theFile = open(fileName, "w")
        pickle.dump(self.value_function, theFile)
        theFile.close()

    def load_value_function(self, fileName):
        theFile = open(fileName, "r")
        self.value_function=pickle.load(theFile)
        theFile.close()

    def agent_message(self,inMessage):
        
        #	Message Description
        # 'freeze learning'
        # Action: Set flag to stop updating policy
        #
        if inMessage.startswith("freeze learning"):
            self.policyFrozen=True
            return "message understood, policy frozen"

        #	Message Description
        # unfreeze learning
        # Action: Set flag to resume updating policy
        #
        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen=False
            return "message understood, policy unfrozen"

        #Message Description
        # freeze exploring
        # Action: Set flag to stop exploring (greedy actions only)
        #
        if inMessage.startswith("freeze exploring"):
            self.exploringFrozen=True
            return "message understood, exploring frozen"

        #Message Description
        # unfreeze exploring
        # Action: Set flag to resume exploring (e-greedy actions)
        #
        if inMessage.startswith("unfreeze exploring"):
            self.exploringFrozen=False
            return "message understood, exploring frozen"

        #Message Description
        # save_policy FILENAME
        # Action: Save current value function in binary format to 
        # file called FILENAME
        #
        if inMessage.startswith("save_policy"):
            splitString=inMessage.split(" ");
            self.save_value_function(splitString[1]);
            print "Saved.";
            return "message understood, saving policy"

        #Message Description
        # load_policy FILENAME
        # Action: Load value function in binary format from 
        # file called FILENAME
        #
        if inMessage.startswith("load_policy"):
            splitString=inMessage.split(" ")
            self.load_value_function(splitString[1])
            print "Loaded."
            return "message understood, loading policy"

        return "SarsaAgent(Python) does not understand your message."



if __name__=="__main__":
    os.system('clear')
    AgentLoader.loadAgent(SarsaAgent())

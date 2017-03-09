import os
import time
import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

class PuddleWorld(Environment):
    WORLD_FREE = 0
    WORLD_OBSTACLE = 5
    WORLD_PUDDLE_DEEP = 1
    WORLD_PUDDLE_DEEPER = 2 
    WORLD_PUDDLE_DEEPEST = 3
    WORLD_GOAL = 4
    randGenerator=random.Random()
    startRow=1
    startCol=1
    epsilon = 0.1
    wind = 0.5
    display_trajectory = False

    currentState=10
    def env_init(self):
        
        self.map=[  [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 2, 2, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 3, 2, 1, 1, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  ]

        #The Python task spec parser is not yet able to build task specs programmatically
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.9 OBSERVATIONS INTS (0 195) ACTIONS INTS (0 3) REWARDS (-3.0 10.0) EXTRA puddle-world by KarthikThiagu."

    def env_start(self):
        start_states = [ (6, 1), (7, 1), (11, 1), (12, 1) ]
        startRow, startCol = start_states[ self.randGenerator.randint( 0, len(start_states) - 1 ) ]
        self.agentRow = startRow
        self.agentCol = startCol

        #print 'The agent begins in state (%d,%d) flattened as %d' % (self.agentRow, self.agentCol, self.calculateFlatState())
        returnObs=Observation()
        returnObs.intArray=[self.calculateFlatState()]

        return returnObs
        
    def env_step(self,thisAction):
        # Make sure the action is valid 
        assert len(thisAction.intArray)==1,"Expected 1 integer action."
        assert thisAction.intArray[0]>=0, "Expected action to be in [0,3]"
        assert thisAction.intArray[0]<4, "Expected action to be in [0,3]"
        
        self.updatePosition(thisAction.intArray[0])

        theObs=Observation()
        theObs.intArray=[self.calculateFlatState()]

        returnRO=Reward_observation_terminal()
        returnRO.r=self.calculateReward()
        returnRO.o=theObs
        returnRO.terminal=self.checkCurrentTerminal()

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):

        if inMessage.startswith("start display-trajectory"):
            self.display_trajectory = True
        if inMessage.startswith("stop display-trajectory"):
            self.display_trajectory = False
        if inMessage.startswith("set-goal-state"):
            _, row, col = inMessage.split(' ')
            self.map[int(row)][int(col)] = self.WORLD_GOAL
        if inMessage.startswith("turn-off-wind"):
            self.wind = 0.0

        return "PuddleWorldEnvironment(Python) does not respond to that message.";

    def checkValid(self, row, col):
        valid=False
        numRows=len(self.map)
        numCols=len(self.map[0])

        if(row < numRows and row >= 0 and col < numCols and col >= 0):
            if self.map[row][col] != self.WORLD_OBSTACLE:
                valid=True
        return valid

    def checkTerminal(self,row,col):
        if self.map[row][col] == self.WORLD_GOAL:
            return True
        return False

    def checkCurrentTerminal(self):
        return self.checkTerminal(self.agentRow,self.agentCol)

    def calculateFlatState(self):
        numCols = len(self.map[0])
        return self.agentRow * numCols + self.agentCol

    def updatePosition(self, theAction):
        # When the move would result in hitting an obstacles, the agent simply doesn't move 
        # newRow and newCol are initialized to the current (row, col)
        newRow = self.agentRow;
        newCol = self.agentCol;
        #print 'Before the action is taken, the state is (%d, %d)' % (newRow, newCol) 
        action_dict = {0 : 'down', 1 : 'up', 2 : 'left', 3 : 'right'}
        #print 'The intended action is %s' % action_dict[theAction]
        # Stochastic actions
        other_actions = set(range(4))
        other_actions.remove(theAction)
        other_actions = list(other_actions)
        if self.randGenerator.random() < self.epsilon:
            theAction = other_actions[ self.randGenerator.randint(0, 2) ]
            #print 'A different action is taken'
        #print 'Actual intended action is %s' % action_dict[theAction]

        if (theAction == 0):#move down
            newRow = self.agentRow + 1;

        if (theAction == 1): #move up
            newRow = self.agentRow - 1;

        if (theAction == 2):#move left
            newCol = self.agentCol - 1;

        if (theAction == 3):#move right
            newCol = self.agentCol + 1;

        #Check if new position is out of bounds or inside an obstacle 
        if(self.checkValid(newRow, newCol)):
            self.agentRow = newRow;
            self.agentCol = newCol;

        # Westerly wind
        if self.randGenerator.random() < self.wind:
            #print 'the wind is present'
            newCol = self.agentCol + 1
            #Check if new position is out of bounds or inside an obstacle 
            if(self.checkValid(newRow, newCol)):
                self.agentRow = newRow;
                self.agentCol = newCol;

        #print 'After the action is taken (%d, %d)' % (self.agentRow, self.agentCol)
    
        if self.display_trajectory:
            time.sleep(0.4)
            os.system('clear')
            self.printState()

    def calculateReward(self):
        if(self.map[self.agentRow][self.agentCol] == self.WORLD_GOAL):
            return 10.0;
        elif(self.map[self.agentRow][self.agentCol] == self.WORLD_PUDDLE_DEEP):
            return -1 * float(self.WORLD_PUDDLE_DEEP);
        elif(self.map[self.agentRow][self.agentCol] == self.WORLD_PUDDLE_DEEPER):
            return -1 * float(self.WORLD_PUDDLE_DEEPER);
        elif(self.map[self.agentRow][self.agentCol] == self.WORLD_PUDDLE_DEEPEST):
            return -1 * float(self.WORLD_PUDDLE_DEEPEST);
        return 0.0;
        
    def printState(self):
        numRows=len(self.map)
        numCols=len(self.map[0])
        print "Agent is at: "+str(self.agentRow)+","+str(self.agentCol)
        print "Col     ",
        for col in range(0,numCols):
            print col%10,
            
        for row in range(0,numRows):
            print
            print "Row: "+str(row%10)+"  ",
            for col in range(0,numCols):
                if self.agentRow==row and self.agentCol==col:
                    print "A",
                else:
                    if self.map[row][col] == self.WORLD_GOAL:
                        print "G",
                    if self.map[row][col] == self.WORLD_PUDDLE_DEEP:
                        print "P",
                    if self.map[row][col] == self.WORLD_PUDDLE_DEEPER:
                        print "R",
                    if self.map[row][col] == self.WORLD_PUDDLE_DEEPEST:
                        print "T",
                    if self.map[row][col] == self.WORLD_OBSTACLE:
                        print "*",
                    if self.map[row][col] == self.WORLD_FREE:
                        print " ",
        print
		

if __name__=="__main__":
    os.system('clear')
    EnvironmentLoader.loadEnvironment(PuddleWorld())


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


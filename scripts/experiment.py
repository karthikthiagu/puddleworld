
import sys
import math
import rlglue.RLGlue as RLGlue

def test():
    RLGlue.RL_agent_message("freeze learning")
    RLGlue.RL_env_message("start display-trajectory")
    RLGlue.RL_episode(0);
    agent_return = RLGlue.RL_return();
    RLGlue.RL_env_message("stop display-trajectory")
    RLGlue.RL_agent_message("unfreeze learning");
    return agent_return

def train():
    for i in range(1000):
        RLGlue.RL_episode(0);
        if (i + 1) % 500 == 0:
            agent_return = test()

RLGlue.RL_init()
train()
RLGlue.RL_agent_message("save_policy results/results.dat")
#RLGlue.RL_agent_message("load_policy results.dat")
#test()


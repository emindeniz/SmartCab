import random
import pandas as pd
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from scipy import stats

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
  
    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.total_reward = 0
        self.completed_trips = 0
        # Here I define my states, actions and initiliaze my Q table.
        states = {'Lights': ['red','red','red','green','green','green'],
                            'next_waypoint': ['forward','right','left','forward','right','left']}  
        self.actions = ['forward','right','left',None]
        self.states = pd.DataFrame(states)        
        #self.actions = pd.DataFrame(actions)
        self.Qtable = np.zeros((6,4))
        self.Qtable.fill(2)
        self.previous_state = 0 #Definition of a previous state is necessary because we never know state we are going to end up with s'
        self.previous_reward = 0
        self.previous_action = 0
        self.time = 1
        self.negative_rewards = []
        self.state = None
      
             

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        print self.next_waypoint

        # TODO: Update state
        states = self.states
        state_criterion = (states['Lights']==inputs['light']) & (states['next_waypoint']==self.next_waypoint)
        current_state = states[state_criterion].index.tolist()[0]
        self.state = states.iloc[current_state].tolist()
        
        # TODO: Learn policy based on state, action, reward
        # Important reinforcement learning parameters
        alpha = 0.8 #learning rate
        gamma = 0 #value for the future value discount
        epsilon = 1/(self.time**(1/4.)) #exploration exploitation dilemma constant, initially close to 1 then approaches 0     
        
        # Update Q table for the previous State action pair, because now we know which state we ended up following the action from the previous step.
        p_state = self.previous_state
        p_reward = self.previous_reward
        p_action = self.previous_action
        self.Qtable[p_state][p_action]=(self.Qtable[p_state][p_action])*(1-alpha)+alpha*(p_reward+gamma*np.amax(self.Qtable[current_state,:]))
        self.previous_state = current_state #Now our current state will become previous state for the next update
        
        # TODO: Select action according to your policy
        #define the probability distribution to either select the best action or to explore
        eps_value = (1,0)
        eps_prob = (epsilon,1-epsilon)
        eps_dist = stats.rv_discrete(name='random_dist', values=(eps_value, eps_prob))
        eps_random = eps_dist.rvs(size=1)[0]
        if eps_random and self.completed_trips<25:
            action_number = random.choice([0,1,2,3])
        else:
            action_number = np.argmax(self.Qtable[current_state,:])
        action = self.actions[action_number]
        self.previous_action = action_number
        
        

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.previous_reward = reward
       
        
        #Keep track of number of completed trips and time
        if reward == 12:
            self.completed_trips +=1
        self.time +=t
        #Keep track of negative rewards collected
        if reward<0:
            self.negative_rewards.append((reward*self.completed_trips))
            
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        print self.negative_rewards

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    print a.Qtable, a.completed_trips

if __name__ == '__main__':
    run()

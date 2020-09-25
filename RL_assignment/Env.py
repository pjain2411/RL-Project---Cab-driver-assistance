# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
#         self.action_space =[[i,j] for i in range(0, m) for j in range(0,m) if i!=j ]
#         self.action_space.append([0,0]) 
        self.action_space = [[i,j] for i in range(5) for j in range(5) if (i!=j) or ((i==0) and (j==0))] 
        self.state_space = [[i,j,k] for i in range(0, m) for j in range(0,t) for k in range(0,d) ]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros(m+t+d)
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)        
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)
            
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):        
        """Takes in state, action and Time-matrix and returns the reward"""
        time_pickup_drop=Time_matrix[action[0]][action[1]][state[1]][state[2]]
        time_driverloc_pickup=Time_matrix[state[0]][action[0]][state[1]][state[2]]
        reward=0
        if((action[0] == 0) & (action[1] == 0)):
            reward=-C
        else:
            reward=(R*time_pickup_drop)-(C*(time_pickup_drop+time_driverloc_pickup))  
        return reward

    def new_datetime(self,state,total_time):
        if state[1]+total_time > 23:
            hour_of_the_day=((state[1]+total_time)%24)
            day_of_week=(state[2] + ((state[1]+total_time)//24))
            if day_of_week > 6:
                day_of_week=(day_of_week % 7)
        else:
            hour_of_the_day=(state[1]+total_time)
            day_of_week=state[2]
        return(int(hour_of_the_day), int(day_of_week))


    def next_state_func(self, state, action, Time_matrix):
        
        """Takes state and action as input and returns next state"""
        next_state=[]
        if (action[0]==0 and action[1]==0):
            total_time=1
            hour_of_the_day,day_of_week = self.new_datetime(state,total_time)
            next_state=[state[0], hour_of_the_day, day_of_week]
        else:
            time_pickup_drop=Time_matrix[action[0]][action[1]][state[1]][state[2]]
            time_driverloc_pickup=Time_matrix[state[0]][action[0]][state[1]][state[2]]
            total_time=time_pickup_drop + time_driverloc_pickup
            hour_of_the_day,day_of_week = self.new_datetime(state,total_time)
            next_state=[action[1], hour_of_the_day, day_of_week]
        return next_state,total_time




    def reset(self):
        return self.action_space, self.state_space, self.state_init

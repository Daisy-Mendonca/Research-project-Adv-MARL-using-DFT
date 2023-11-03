'''
title      :   environment.py
version    :   0.0.1
date       :   30.03.2023 12:01:31
fileName   :   environment.py
author     :   Joachim Nilsen Grimstad
contact    :   Joachim.Grimstad@ias.uni-stuttgart.de

description:   None

license    :   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
               For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  

disclaimer :   Author takes no responsibility for any use.
'''

# Dependencies
import functools
from os import system
import time 
import numpy as np


import gymnasium
from gymnasium.spaces import Discrete, MultiBinary
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

MOVES = ["Move", "Not_Move"]

class ESREL_Example(AECEnv):                                                                                         
    'ESREL Example dynamic fault tree game'
    metadata = {"render_modes": ["human"], "name": "rps_v2"}                                                                                                
    
    # Init
    def __init__(self, system, game, render_mode=None):
        super().__init__()                                                                                         
        # Input Parameters  
        # if not isinstance(mission_time, int): 
        #     raise TypeError("mission_time must be an integer")
        
        # Stable_baselines3 Test
        self.reward_range = (-np.inf, np.inf)

        # Initial Parameters
        self.system = system
        self.game_object = game
        # self.mission_time = mission_time
        self.initial_resources = game.initial_resources
        self.resources = game.initial_resources
        self.render_mode = render_mode
        self.clock = 0
        self.num_moves = 0
        self.system_state = self.system.state
        self.NUM_ITERS = game.get_max_steps() 
        self.done = False
        # Agents
        self.agents =  game.get_players()
        self.possible_agents = self.agents.copy()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.n_agents = game.get_num_players()
        self.agents_good = game.get_players_strategy_type("GOOD")
        self.agents_bad = game.get_players_strategy_type("BAD")
        self.n_agents_good = game.get_num_players_strategy_type("GOOD")
        self.n_agents_bad  = game.get_num_players_strategy_type("BAD")

        # Spaces
        #self.action_spaces = {"agent_red": Discrete(13), "agent_blue": Discrete(13)}          
        self.action_spaces = {agent: Discrete(13) for agent in self.possible_agents}                      
        self.observation_spaces = {agent: MultiBinary(1) for agent in self.possible_agents} 
        self.observations = {agent: self.game_object.get_event_states() for agent in self.agents} 
        # print(self.observations)

        # I don't even know anymore
        #self.rewards = {"agent_red": 0, "agent_blue": 0}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}                                                                                                       
        self.infos = {agent: dict() for agent in self.agents}                                                                 
        self.terminations = {agent: False for agent in self.agents}                                                                    
        self.truncations = {agent: False for agent in self.agents}

##########################################################################################################################

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent): # returns observation of an agent
        #print(np.array(self.observations[agent]))
        return np.array(self.observations[agent])

    def reset(self, seed=None, return_info=False, options=None):
        self.has_reset = True
        self.agents = self.possible_agents[:]
        self.resources = self.initial_resources
        self.clock = 0
        self.game_object.reset_game()
        # self.system_state = self.system.state
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.state = {agent: self.game_object.get_event_states() for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.action_spaces = {agent: Discrete(13) for agent in self.possible_agents}                      
        self.observation_spaces = {agent: MultiBinary(1) for agent in self.possible_agents} 
        self.observations = {agent: self.game_object.get_system_state() for agent in self.agents} 
        if self.render_mode == "human":
            self.render()
        return self.observations
        # return self.system.obs

    def step(self, action):

        # Removes terminated or truncated agents from 'everything'.
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]): # cycle is not terminated
            return self._was_dead_step(action)

        # Selects agent for this step
        agent = self.agent_selection

        # Action Mask - Poor implementation?, they've done it differently in the PettingZoo Chess documentation where a mask is returned along with the observations in the observe function.  
        action, cost = self.game_object.action_masking(agent, action)
        #print(self.game_object.get_mask(agent))
        self.resources[agent] = self.resources[agent] - cost
        #print(self.resources[agent])


        #if action == 0: # If Action impossible, do no action. Might want to count illegal actions here and add to information.
        #    action = 0
        #    cost = 0
        #else: # Else check if I can afford action.
        #    cost = self.game_object._get_action_cost()
            #print(cost[action])
        #    if cost > self.resources[agent]: # If I cannot afford action, do no action.
        #        action = 0
        #        cost = 0
        # If I can afford it, do action and pay cost.
        self.game_object.apply_action(agent, action, agent.player_type)
        self.game_object.set_event_states()
        #print(agent.player_type)
        #self.resources[agent] = self.resources[agent] - cost

        # Update System
        self.system_state = self.game_object.time_step()
        #print(self.system_state)
        self.clock += 1
        #print(self.game_object.get_event_states())

        # update observation of next agent
        self.observations[self.agents[1 - self.agent_name_mapping[agent]]] = self.game_object.get_system_state()
        #print(self.observations)
##########################################################################################
#                                   REWARDS I HATE THEM
##########################################################################################
        if self._agent_selector.is_last():      
            self._cumulative_rewards[agent] = 0 # Uncomment if we dont want to cumulate rewards, depends on the algorithm...

            for agent in self.agents:
                self.rewards[agent] = self.calculate_rewards(agent)

            self.clock -= 1
            self.num_moves += 1
##########################################################################################

            # Update termination – game is "won"
            if self.system_state == 0:
                self.done = True
                self.terminations = {agent: True for agent in self.agents}

            # Update truncation - time is over.
            self.truncations = {
                agent: self.num_moves >= self.NUM_ITERS for agent in self.agents}
    
            # Infos
            self.infos = {agent: {} for agent in self.agents}

            # DEBUGGING!
            #self.infos = {"agent_red": {'time' : self.clock, 'state' : self.system_state}, "agent_blue": {'time' : self.clock, 'state' : self.system_state}}
            self.game_object.increase_steps()
            self.timestep = self.game_object.get_current_step() 
            self.game_object.is_game_over()
        # Next Agent
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()
        return self.observations, self.rewards, self.done, self.infos
###########################################################################    

    def calculate_rewards(self, agent):
        reward = 0
        num_system_objects = self.game_object.get_num_system_objects()
        for i in range(0, num_system_objects):
            if agent.player_type == 'BAD':
                if self.system_state[i] == 1:
                    reward -= 1
                elif self.system_state[i] == 0:
                    reward += 1
                    agent.increase_score()
                    #print("BAD")
                    #print(agent.get_score())
            elif agent.player_type == 'GOOD':
                if self.system_state[i] == 1:
                    reward += 1
                    agent.increase_score()
                    #print("GOOD")
                    #print(agent.get_score())
                elif self.system_state[i] == 0:
                    reward -= 1
            #print(self.system_state)
            return reward


    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def render(self, mode = "human"):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)
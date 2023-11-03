'''
title		:	Dynamic Fault Tree Game
version     :   0.0.1
date		:	27.03.2023
fileName	:	system.py
author		:	Joachim Nilsen Grimstad
contact 	:   Joachim.Grimstad@ias.uni-stuttgart.de
description :   System

license 	:   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  
disclaimer	:   Author takes no responsibility for any use.
'''

# Imports
from sre_parse import State
from dft import No_Action
from dft import Basic_Event
    
# Classes

class System:
    'System Model Objects'
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.objects = []
        self.clock = 0

    def get_name(self):
        return self.name
    
    def get_object(self, object_name):
        'returns the object with a given object_name'
        for obj in self.objects:
            if obj.name == object_name:
                return obj

class System_DFT(System):
    'Dynamic Fault Tree System Model Object'
    def __init__(self, name):
        System.__init__(self, name, 'DFT')

#     # ___________________________Methods_______________________________
    
    def reset_system(self):
        'resets system'
        self.clock = 0
        for obj in self.objects:
            obj.reset()
        self.top_event.update_state()
        self.state = self.top_event.state

    def instantiate(self):
        'instantiates the system, updates the states of all the events based on basic events.'
        self.top_event = self.get_top_event()
        self.reset_system()
        self._set_actions()
        self._set_observations()
        
    def get_top_event(self):
        'returs the top event'
        for obj in self.objects:
            if obj.type == 'TOP':
                return obj

    def _set_actions(self):
        no_action = No_Action('No Action')
        actions = []
        actions.append(no_action)
        for obj in self.objects:
            if obj.type =='BASIC':
                actions.append(obj)
        self.actions = actions
        #print(len(actions))
        return actions

    def num_actions(self):
        return len(self._set_actions())
    
    def get_num_event_states(self):
        return len(self.get_event_state())
    
    def get_num_repair_events(self):
        return len(self.get_repair_status())

    def _set_observations(self):
        observations = []
        for obj in self.objects:
            if obj.type == 'BASIC':
                observations.append(obj)
        self.observations = observations
        return observations
    
    def num_observation(self):
        #print(len(self._set_observations()))
        return len(self._set_observations())

    def apply_action(self, agent, action, stratergy_type):
        #print(stratergy_type)
        if stratergy_type == 'BAD':
            self.actions[action].red_action()
        elif stratergy_type == 'GOOD':
            self.actions[action].blue_action()

    def time_step(self):
        for obj in self.observations:
            obj.time_step()
        state = self.update_system()
        self.clock += 1 
        return state
        
    def update_system(self):
        self.top_event.update_state()
        self.state = self.top_event.state
        return self.state
    
    def get_repair_status(self):
        status = []
        for obj in self.objects:
            if obj.type == 'BASIC':
                status.append(obj.get_repair_status())
        return status
    
    def get_event_state(self):
        states = []
        for obj in self.objects:
            if obj.type == 'BASIC':
                states.append(obj.get_event_state())
        return states
    
    def get_action_cost(self, action):
        costs = []
        costs.append(0) # for no action
        for obj in self.objects:
            if obj.type == 'BASIC':
                costs.append(obj.get_failure_cost())
        #print(costs[action])
        return costs[action]

    def get_all_events(self):
        top_event_array = []
        intermediate_events_array = []
        basic_events_array = []
        for obj in self.objects:
            if obj.type == 'BASIC':
                basic_events_array.append(obj)
            elif obj.type == 'INTERMEDIATE':
                intermediate_events_array.append(obj)
            elif obj.type == 'TOP':
                top_event_array.append(obj)
        return top_event_array, intermediate_events_array, basic_events_array
    

'''
title      :   dft.py
version    :   0.0.1
date       :   17.04.2023 10:54:19
fileName   :   dft.py
author     :   Joachim Nilsen Grimstad
contact    :   Joachim.Grimstad@ias.uni-stuttgart.de

description:   None

license    :   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  

disclaimer :   Author takes no responsibility for any use.
'''

class Element:
    'Abstract Base Class for model Elements'
    def __init__(self, name, type):
        super().__init__()
        self.name = name
        self.type = type
        self.state = -1

class Event(Element):
    'Abstract Base Class for non-basic events'

    def __init__(self, name, gate_type, type):
        Element.__init__(self, name, type)
        self.input = []
        self.gate_type = gate_type
        self.using_spare = 0
        
    def update_state(self):
        for obj in self.input: # Updates all the input branches
            obj.update_state()
        if self.gate_type == 'AND': # If all inputs are 0, then state = 0, otherwise, state = 1
            if sum([obj.state for obj in self.input]) == 0:
                self.state = 0
            else:
                self.state = 1
        elif self.gate_type == 'OR': # If any input is != 0, then state = 0, otherwise, state = 1
            for obj in self.input:
                if obj.state != 1:
                    self.state = 0
                    return
            self.state = 1
        elif self.gate_type == 'FDEP': # FDEP gates only accepts one input precedence, must combine with and or events prior to input signal.
            self.state = self.input[0].state
        elif self.gate_type == 'CSP': # Cold Spare currently, only accepts 1 compeditor, and 1 spare. Also the 'main' input must come first in the input list.
            self.main_functioning = self.input[0].state # Is main functioning?
            self.spare_functioning = self.spare.state # Is Spare functioning?     
            if self.main_functioning == 1:
                self.state = 1
                self.using_spare = 0
            elif self.main_functioning == 0 and self.compeditor.using_spare == 0 and self.spare_functioning == 1:
                self.state = 1
                self.using_spare = 1
            else:
                self.state = 0
                self.using_spare = 0

    def reset(self):
        pass

    
class Basic_Event(Element):
    'Basic Event class'
    def __init__(self, name, mttr, repair_cost, failure_cost, initial_state):
        Element.__init__(self, name, 'BASIC')
        self.mttr = int(mttr)
        self.repair_cost = int(repair_cost)
        self.failure_cost = float(failure_cost)
        self.initial_state = int(initial_state)
        
        # Attributes
        self.state = self.initial_state
        self.repairing = 0
        self.remaining_time_to_repair = 0

    def reset(self):
        self.state = self.initial_state
        self.repairing = 0
        self.remaining_time_to_repair = 0

    def update_state(self):
        'states for basic events are manipulated through actions, and reset through reset'
        pass

    def red_action(self):
        'activate basic event'
        self.state = 0
        self.remaining_time_to_repair = self.mttr

    def blue_action(self):
        'inactivate basic event'
        self.repairing = 1

    def time_step(self):
        if self.state == 1:
            pass
        elif self.state == 0 and self.repairing == 1 and self.remaining_time_to_repair > 1:
            self.remaining_time_to_repair += -1
        elif self.state == 0 and self.repairing == 1 and self.remaining_time_to_repair == 1:
            self.state = 1
            self.repairing = 0
            self.remaining_time_to_repair = 0
        elif self.state == 0 and self.repairing == 0:
            pass
        
    
    def get_repair_status(self):
        return self.repairing
    
    def get_event_state(self):
        return self.state
    
    def get_failure_cost(self):
        return self.failure_cost

class No_Action(Element):
    'No Action class' 

    def __init__(self, name):
        Element.__init__(self, name, 'No Action')
        
    def red_action(self):
        'skip turn'
        pass

    def blue_action(self):
        'skip turn'
        pass
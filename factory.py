'''
title		:	Dynamic Fault Tree Game
version     :   0.0.1
date		:	27.03.2023
fileName	:	factory.py
author		:	Joachim Nilsen Grimstad
contact 	:   Joachim.Grimstad@ias.uni-stuttgart.de
description :   Factory

license 	:   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  
disclaimer	:   Author takes no responsibility for any use.
'''

# Imports
import xml.etree.ElementTree as ET
from system import System, System_DFT
from dft import *

class Factory:
    '''
    Anstract Factory Class
    '''

    def get_system_and_builder(data):
        'Returns the correct factory and system object'
        if data['tag'] == 'model':
            system_type = data['type']
            if system_type == 'DFT':
                factory = Factory_DFT
            # Other Concrete Factories goes here.
            system = factory.create_system(data)
        return factory, system


class Factory_DFT:
    '''
    Factory for Dynamic Fault Tree
    '''

    def create_system(data):
        system = System_DFT(data['name'])
        return system

    def create_object(data, system):
        'creates or manipulates the different objects in the system'
        obj_tag = data['tag']
        if obj_tag == 'model':
            pass
        elif obj_tag == 'event':
            if data['type'] == 'TOP':
                obj = Event(data['name'], data['gate_type'], data['type'])
                system.objects.append(obj)
            elif data['type'] == 'INTERMEDIATE':
                obj = Event(data['name'], data['gate_type'], data['type'])
                system.objects.append(obj)
            elif data['type'] == 'BASIC':
                obj = Basic_Event(data['name'], data['mttr'], data['repair_cost'], data['failure_probability'], data['initial_state'])
                system.objects.append(obj)
        elif obj_tag == 'precedence':            
            source_name = data['source']
            target_name = data['target']
            precedence_type = data['type']
            source = system.get_object(source_name)
            target = system.get_object(target_name)
            target.input.append(source)
            if precedence_type == "CSP":
                compeditor_name = data['compeditor']
                compeditor = system.get_object(compeditor_name)
                target.compeditor = compeditor
                target.spare = source

    def instantiate_system(system):
        system.instantiate()






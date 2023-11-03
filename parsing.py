'''
title		:	Dynamic Fault Tree Game
version     :   0.0.1
date		:	27.03.2023
fileName	:	parsing.py
author		:	Joachim Nilsen Grimstad
contact 	:   Joachim.Grimstad@ias.uni-stuttgart.de
description :   Parser

license 	:   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  
disclaimer	:   Author takes no responsibility for any use.
'''



# Imports 
import xml.etree.ElementTree as ET
from system import System
from factory import Factory

class Parse:
    '''
    Simple model parse interface
    '''
    # init method or constructor
    def __init__(self,name):
        self.name = name

    def from_file(self, file_name):
        file_type = file_name.split('.')[-1]                    # Gets the file_type 
        if file_type == 'xml':      
            Parser = Parse_XML(name="Parser")                            # If xml
            system = Parser.from_file(file_name)             # Use xml parser
            return system
        
        # Other file_type parsers are called here.

class Parse_XML():
    '''
    XML parse interface
    '''
    def __init__(self,name):
        self.name = name

    def from_file(self, file_name):                                   
        tree = ET.parse(file_name)
        root = tree.getroot()
        factory, system = Factory.get_system_and_builder({"tag": root.tag, **root.attrib})  # Uses Factory interface to return concrete factory interface, and system object
        for branch in root.iter():
            factory.create_object({"tag": branch.tag, **branch.attrib}, system)             # Uses the concrete factory interface to create objects 
        system.instantiate()                                                                # Uses the concrete factory to instantiate the model
        return system

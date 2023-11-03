from parsing import Parse
from game import Game
from environment import ESREL_Example
import json

AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}

def env_creator(model_file, agent_type, state_init):
    Parser = Parse(name="Parser")
    
    dft = Parser.from_file(model_file)
    # print(dft)
    dft.reset_system()

    game = Game([dft], 4, 25)

    if agent_type == AGENTS_TYPE[0]:
       game.create_player("agent_blue", "GOOD", 1000)  
       game.create_player("agent_black", "GOOD", 1000)
    
    elif agent_type == AGENTS_TYPE[1]:
       game.create_player("agent_red", "BAD", 1000)
       game.create_player("agent_white", "BAD", 1000)
       
    elif agent_type == AGENTS_TYPE[2]:
       game.create_player("agent_red", "BAD", 1000)
       game.create_player("agent_blue", "GOOD", 1000)  

       env = ESREL_Example(dft,game)
   
    return env, game
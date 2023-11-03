#Imports
from environment import ESREL_Example
from system import System 
from parsing import Parse
from game import Game

from ppo import train, create_ppo_agent
from env_creator import env_creator



# Parser = Parse(name="Parser")
# dft = Parser.from_file("model.xml")   #system object
# game = Game([dft], 2, 1000)
# env = ESREL_Example(10, dft, game, 1000)

def Training(PATH, env, total_episodes):

 '''Training'''
 
 PATH = PATH 
 
 actor_model = PATH + '/ppo_actor_'
 critic_model =  PATH + '/ppo_critic_'
 train(PATH, env, actor_model, critic_model, total_episodes = total_episodes)

INPUT_MODELS = {0:'model.xml'}
AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}

def Simulating(PATH_SOURCE, PATH_SINK, env, game, simulations):
     
 '''Simulating'''
 
 good_ai = create_ppo_agent(env, PATH_SOURCE + '/ppo_actor_',  PATH_SOURCE + '/ppo_critic_')
 bad_ai  = create_ppo_agent(env, PATH_SOURCE + '/ppo_actor_',  PATH_SOURCE + '/ppo_critic_')
 for i in range(simulations):
     game.reset_game()
     print("Simulation for " + game.get_system_objects()[0].get_name()+ ":" + str(i))
     game.play(PATH_SINK, good_ai=good_ai,bad_ai=bad_ai)
     print("\n----------------------------------------------------\n")


if __name__ == "__main__":
    env,game = env_creator('./' + INPUT_MODELS[0], AGENTS_TYPE[2], 0) 
    #PATH_PPO = './PPO_Agent/' + game.get_system_objects()[0].get_name()
    PATH_PPO = './PPO_Agent/'
    Training(PATH_PPO, env, total_episodes = 10)

    
    # PATH_PPO = './PPO_Agent/' + game.get_system_objects()[0].get_name() 
    # PATH_Records = './Game_Records/' + '/' 
    # Simulating(PATH_PPO, PATH_Records, env, game, simulations=10)
    # print("\n------------------------\n")

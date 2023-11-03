
import random
import os
import csv
import numpy as np
from bachify import *

VALID_ACTIONS = []

class Game:
    def __init__(self, System_Objects, max_num_players, max_steps):
        self.system_objects = System_Objects
        self.max_num_players = max_num_players
        self.players = []
        self.event_states = []
        self.repair_status = []
        self.num_players = len(self.players)
        self.current_step  = 0
        self.max_steps = max_steps
        self.game_over_flag = False 
        self.players_good = []
        self.players_bad  = []
        self.initial_resources = {}
        self.resources = {}
        self.initial_state = self.set_event_states()
        self.current_state = self.initial_state
        # print(self.system_objects)

    def _set_actions(self):
        actions = []
        for system_object in self.system_objects:
            for action in system_object._set_actions():
                if(actions.__contains__(action) == 0):
                    actions.append(action)
                    VALID_ACTIONS.append(action)
            #print(actions)
            return actions
        
    def _set_observations(self):
        observations = []
        for system_object in self.system_objects:
            observations.append(system_object._set_observations())    
        return observations
    
    def num_observation(self):
        for system_object in self.system_objects:
            num_observations = system_object.num_observation()
        return num_observations

    def apply_action(self, agent, action, stratergy_type):
        #print(agent.player_type)
        for system_object in self.system_objects:
            system_object.apply_action(agent, action, stratergy_type)

    
    def create_player(self, name, strategy_type, resources):
        if len(self.players) < self.max_num_players:  
            player = Player(name, strategy_type, resources)
            self.initial_resources[player] = resources
            self.resources[player] = resources
            self.players.append(player)
            if strategy_type == "GOOD":
                self.players_good.append(player)
            elif strategy_type == "BAD":
                self.players_bad.append(player)
            return player
        else:
            print("Maximum number of players has been reached")
            return None
        
    def is_game_over(self):
        if self.current_step == self.max_steps:
           self.game_over_flag = True 
    
    def get_system_state(self):
        states = []
        for system_object in self.system_objects:
            states.append(system_object.update_system())
        return states
    
    def get_winner(self):
        max_score = -1000
        winners = [] 
        for player in self.players:
            if player.get_score() >  max_score:
                max_score = player.get_score()
                
        for player in self.players:
            if player.get_score() ==  max_score:
                winners.append(player)
        return winners
    
    def get_player_by_name(self, name):
        for player in self.players:
            if player.get_name() == name:
                return player
        raise ValueError("There is no player with such name")

    def get_players(self):
        return self.players
    
    def get_game_over_flag(self):
        return self.game_over_flag
    
    def get_current_step(self):
        return self.current_step
    
    def get_max_steps(self):
        return self.max_steps
    
    def get_system_objects(self):
        return self.system_objects
    
    def get_num_system_objects(self):
        return len(self.system_objects)
    
    def increase_steps(self):
        self.current_step  = self.current_step + 1
    
    def set_event_states(self):
        new_states = []
        for system_object in self.system_objects:
            new_states.append(system_object.get_event_state())
        self.current_state = new_states

    def get_event_states(self):
        event_states = []
        for system_object in self.system_objects:
            # print(self.system_objects)
            event_states.append(system_object.get_event_state())
        #print(event_states)
        return event_states
    
    def get_repair_status(self):
        repair_status = []
        for system_object in self.system_objects:
            repair_status.append(system_object.get_repair_status())
        #print(repair_status)
        return repair_status

    def get_num_actions(self):
        #for system_object in self.system_objects:
        #    num_actions = system_object.num_actions()
        #print(len(self._set_actions()))
        return len(self._set_actions())
    
    def get_num_event_states(self):
        for system_object in self.system_objects:
            num_event_states = system_object.get_num_event_states()
        return num_event_states
    
    def get_num_repair_events(self):
        for system_object in self.system_objects:
            num_repair_events = system_object.get_num_repair_events()
        return num_repair_events

    def get_mask(self, player):
        mask = []
        num_actions = self.get_num_actions()
        event_states = self.get_event_states()
        repair_status = self.get_repair_status()
        num_system_objects = self.get_num_system_objects()
        # print((event_states[0][12]))
        # print((repair_status[0]))
        # print(num_system_objects)
        for i in range (0, num_system_objects):
            mask.append(1)
            for j in range (0, num_actions-1):
                mask.append(player.update_valid_actions_mask(event_states[i][j], repair_status[i][j]))
                #print(player.update_valid_actions_mask(event_states[i][j], repair_status[i][j]))
        #print(mask)
        return mask
    
    def _get_action_cost(self, action):
        action_costs = []
        for system_object in self.system_objects:
            action_costs.append(system_object.get_action_cost(action))
            #print(action_costs)
        return action_costs
    
    def get_players_strategy_type(self, strategy_type):
        if strategy_type == "GOOD":
           return self.players_good
        elif strategy_type == "BAD":
           return self.players_bad
        raise ValueError("There is no player with such strategy type")
    
    def get_num_players(self):
        return len(self.players)
    
    def get_num_players_strategy_type(self, strategy_type):
        if strategy_type == "GOOD":
           return len(self.players_good)
        elif strategy_type == "BAD":
           return len(self.players_bad)
        raise ValueError("There is no player with such strategy type")
    
    def time_step(self):
        states = []
        for system_object in self.system_objects:
            states.append(system_object.time_step())
        return states

    def get_resources_players(self):
        resources_players = {}
        for player in self.players:
            resources_players[player] = player.get_resources()
        return resources_players
    
    def play(self, PATH, good_ai, bad_ai):
        self.current_step = 0
        states = []
        PATH = PATH + 'CSV/player_behavior.csv'
        while (not self.game_over_flag):    

            for player in self.players:
                observation = self.get_system_state()
                #action = player.choose_action(self._set_actions)  # temporary funtion
                if player.get_player_strategy() == 1:
                    model = good_ai
                else:
                    model = bad_ai
                obs = batchify_obs(observation, player, model.device)
                action_mask = batchify(self.get_mask(player), model.device)
                action,_,_,_ = model.get_action_and_value(obs, player.get_player_strategy_name(), invalid_action_masks = action_mask ) #yet to implement
                player_action = unbatchify(action)[0]
                player_action, _ = self.action_masking(player, player_action)
                self.apply_action(player, action, player.player_type)
                states.append(self.get_system_state())
                print("player.player_type...............",player.player_type)
                if(player.player_type == 'agent_red'):
                    if (states == 0):
                        player.increase_score()
                elif (player.player_type == 'agent_blue'):
                    if (states == 1):
                        player.increase_score()
                #self.record_player_behavior(PATH,  player, observation, player_action)
                self.increase_steps()
                print("I'm here........")
            self.is_game_over()
        winners = self.get_winner()
        print("the winner is ", winners)


    def action_masking(self, player, action):
        # Action Mask - Poor implementation?, they've done it differently in the PettingZoo Chess documentation where a mask is returned along with the observations in the observe function.  
        mask = self.get_mask(player)
        num_system_objects = self.get_num_system_objects()
        # print(mask)
        # print(self.resources[agent])
        if mask[action] == 0: # If Action impossible, do no action. Might want to count illegal actions here and add to information.
            action = 0
            cost = 0
            player.set_took_invalid_action(True)
            return action, cost
        else: # Else check if I can afford action.
            cost = self._get_action_cost(action)
            #print(cost[0])
            for i in range (0,num_system_objects):
                if cost[i] > self.resources[player]: # If I cannot afford action, do no action.
                    action = 0
                    cost = 0
                    player.set_took_invalid_action(True)
                else:
                    player.set_took_invalid_action(False)    
            self.resources[player] = self.resources[player] - cost[i]
            player.set_resources(self.resources[player])
        
            return action, cost[i]
    
    def record_player_behavior(self, path,  player, observation, action_mask): 
        print(action_mask)
        print(player.get_valid_actions())
        action = player.get_valid_actions()[action_mask]
        header = ['timestamp', 'player_name', 'player_obs', 'player_res', 'player_action', 'top_event', 'player_score']    
        data   = [self.current_step, player.get_name(), observation, player.get_resources(), action, self.get_system_state(), player.get_score()]
        isExist = os.path.exists(path)
        if (not isExist):
            f = open(path, 'a+', newline='', encoding='utf-8')
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data) 
        else:
            f = open(path, 'a+', newline='', encoding='utf-8')
            writer = csv.writer(f)
            writer.writerow(data) 

    def reset_game(self):
        for system_object in self.system_objects:
          system_object.reset_system()
          
        for player in self.players:
           player.reset_player()
           self.resources[player] = self.initial_resources[player]
               
        # self.current_state = self.initial_state
        self.current_step  = 0  
        self.game_over_flag = False
    

class Player:
    def __init__(self, name, player_type, resources):
        self.name = name
        self.valid_actions = []
        self.valid_actions_mask = []
        self.masks = {"Not_Active" : 0, "Active": 1}
        self.valid_actions_costs = []
        self.score = 0
        self.player_type = player_type
        self.initial_resources = resources
        self.resources = self.initial_resources
        self.took_invalid_action = False
        self.strategy_type  = player_type
    
    def increase_score(self):
        self.score = self.score + 1

    def get_valid_actions(self): 
        return self.valid_actions
    
    def get_num_valid_actions(self): 
        return len(self.valid_actions)
    
    def get_score(self):
        return self.score
    
    def choose_action(self, valid_actions):
        return valid_actions[random.randint(0,len(valid_actions) - 1)]

    def get_name(self):
        return self.name
    
    def get_player_type(self):
        if self.player_type == 'GOOD':
            return 'agent_red'
        elif self.player_type == 'BAD':
            return 'agent_blue'

    def reset_player(self):
        self.score = 0

    def add_valid_action(self, action, cost):
        #print(VALID_ACTIONS)
        if action in VALID_ACTIONS:
            self.valid_actions.append(action)
            self.valid_actions_mask.append(self.masks["Active"])
            self.valid_actions_costs.append(cost)
        else:
            raise ValueError("Not valid action") 
        
    def update_valid_actions_mask(self, state, repairing): 
            player_type = self.get_player_type()
            if  (player_type == 'agent_blue'):
                if (repairing == 1):
                    self.valid_actions_mask = (self.masks["Not_Active"])
                else:
                   self.valid_actions_mask = (self.masks["Active"])
            elif (player_type == 'agent_red'):
                if (repairing == 1 and state == 0):
                    self.valid_actions_mask = (self.masks["Not_Active"])
                else:
                   self.valid_actions_mask = (self.masks["Active"]) 
            #print(player_type)
            return self.valid_actions_mask     

    def get_player_strategy(self):
        if self.strategy_type == 'GOOD':
            return 1
        elif self.strategy_type == 'BAD':
            return -1
        return 0
    
    def get_player_strategy_name(self):
        return self.strategy_type

    def get_valid_actions_mask(self): 
        return self.valid_actions_mask
    
    def set_took_invalid_action(self,value):
        self.took_invalid_action = value

    def get_took_invalid_action(self):
        return self.took_invalid_action
    
    def reset_player(self):
        self.score = 0
        self.resources = self.initial_resources

    def get_resources(self):
        return self.resources
    
    def set_resources(self, resources):
        self.resources = resources
            
def batchify_obs(obs, agent, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = {agent: obs}
    obs = np.stack([obs[a] for a in obs], axis=0)
    # print(obs)
    # transpose to be (batch, channel, height, width)
    # obs = obs.transpose(0, -1, 1, 2)
    # print(obs)
    # convert to torch
    obs = torch.tensor(obs).to(device)
    
    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def unbatchify(x):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    # x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x[0]    
    
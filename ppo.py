# import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from env_creator import env_creator
import matplotlib.pyplot as plt
from bachify import batchify, batchify_obs, unbatchify
gpu = False
device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
INPUT_MODELS = {0:'model.xml'}
model_file = './' + INPUT_MODELS[0]
AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, strategy_type):
        self.strategy_type = strategy_type
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.actor = nn.Linear(1024, out_dim)
        self.critic = nn.Linear(1024, 1)

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, env, in_dim, out_dim):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available() and gpu) else "cpu")
        self.env = env
        self.NeuralNetworks = {}
        self.NeuralNetworks["GOOD"] = NeuralNetwork(in_dim, out_dim, "GOOD").to(self.device)
        self.NeuralNetworks["BAD"]  = NeuralNetwork(in_dim, out_dim, "BAD").to(self.device)

    def get_value(self,  x, strategy_type):
        hidden = self.NeuralNetworks[strategy_type].network(x / 1.0)
        return self.NeuralNetworks[strategy_type].critic(hidden)
      

    def get_action_and_value(self, x, strategy_type, action=None, invalid_action_masks=None):
        #print(x)
        #print(".......................")
        #print(invalid_action_masks)
        #print("|||||||||||||||||||||||||||||||||||||||")
        hidden = self.NeuralNetworks[strategy_type].network(x / 1.0)
        logits = self.NeuralNetworks[strategy_type].actor(hidden)
        split_logits = torch.split(logits,  1)
        
        if invalid_action_masks is not None:
            # print("Not None")
            split_invalid_action_masks = torch.split(invalid_action_masks, 1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        else:
            # print("None")
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
       
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.NeuralNetworks[strategy_type].critic(hidden)


    def learn(self, PATH, env, total_episodes):     
        """ALGO PARAMS"""
        ent_coef = 0.1
        vf_coef = 0.1
        clip_coef = 0.1
        gamma = 0.99
        batch_size = 32
        
        # Train the actor and critic networks. Here is where the main PPO algorithm resides.
        print("Learning...")
        
        """ ENV SETUP """
        max_cycles = env.game_object.get_max_steps() + 3
        num_agents = len(env.possible_agents)
        stack_size = env.game_object.get_num_system_objects()
        depth = env.game_object.num_observation()
        indeces = ["GOOD", "BAD"]
        """ ALGO LOGIC: EPISODE STORAGE"""
        end_step = 0
        total_episodic_return = 0 
        
        rb_obs = {}
        rb_actions = {}
        rb_logprobs = {}
        rb_rewards = {}
        rb_terms = {}
        rb_values = {}
        rb_invalid_action_masks = {}
        for i in range(len(indeces)):
            index = indeces[i]
            rb_obs[index] = torch.zeros((max_cycles, num_agents, stack_size)).to(device)
            rb_actions[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device) 
            rb_logprobs[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)   
            rb_rewards[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)   
            rb_terms[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)    
            rb_values[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)
            agent = env.game_object.get_players_strategy_type(index)[0]
            rb_invalid_action_masks[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index)) + (depth + 1,)).to(device)
 
        """ LEARNER SETUP """
        optimizer = {}
        for i in range(len(indeces)):
            index = indeces[i]
            optimizer [index] = optim.Adam(self.NeuralNetworks[index].parameters(), lr=0.001, eps=1e-5)

        """ Algorithm behaviour """
        policy_losses = {}
        value_function_loss = {}
        explained_variance = {}
        for i in range(len(indeces)):
            index = indeces[i]
            policy_losses[index] = []
            value_function_loss[index] = []
            explained_variance[index] = []

        """ TRAINING LOGIC """
        # train for n number of episodes
        for episode in range(total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                env.reset(seed=None) 
                # reset the episodic return
                total_episodic_return = {}
                for i in range(len(indeces)):
                    index = indeces[i]
                    total_episodic_return[index] = 0
                # each episode has num_steps
                for step in range(0, max_cycles):
                    # rollover the observation
                    actions = {}
                    logprobs= {}
                    values  = {}
                    rewards = {}
                    terminations = {}
                    invalid_action_masks = {}
                    for i in range(len(indeces)):
                        index = indeces[i]
                        actions[index] = {}
                        logprobs[index] = {}
                        values[index] = {}
                        rewards[index] = {}
                        terminations[index] = {}
                        invalid_action_masks[index] = {}
                    for agent in env.agents:
                        observation, reward, termination, truncation, info = env.last()
                        if termination or truncation:
                           action = None
                        else:
                            #invalid_action_masks[agent.get_player_strategy_name()][agent] = env.game_object.get_mask(agent)
                            #obs = batchify_obs(observation, agent, device)
                            #action_mask = batchify(invalid_action_masks[agent.get_player_strategy_name()][agent],device)
                            #print(observation)
                            invalid_action_masks[agent.get_player_strategy_name()][agent] = env.game_object.get_mask(agent)
                            obs = batchify_obs(observation, agent, device)
                            action_mask = {agent:env.game_object.get_mask(agent)}
                            action_mask = batchify(action_mask,device)
                            action, logprob, _, value = self.get_action_and_value(obs, agent.get_player_strategy_name(), invalid_action_masks = action_mask)
                            action = unbatchify(action)[0]
                            logprob = unbatchify(logprob)
                            value = unbatchify(value)
                            actions[agent.get_player_strategy_name()][agent] = action
                            logprobs[agent.get_player_strategy_name()][agent] = logprob
                            values[agent.get_player_strategy_name()][agent] = value
                        env.step(action)  
                    if (action != None):
                       for agent in env.agents:
                           rewards[agent.get_player_strategy_name()][agent] = env.rewards[agent] 
                           terminations[agent.get_player_strategy_name()][agent] = env.terminations[agent] 
                       for i in range(len(indeces)):
                           index = indeces[i]
                           rb_obs[index][step] = batchify(env.observations, device)
                           rb_rewards[index][step] = batchify(rewards[index], device)
                           rb_terms[index][step] = batchify(terminations[index], device)
                           rb_actions[index][step] = batchify(actions[index],device)
                           rb_logprobs[index][step] = batchify(logprobs[index],device)
                           rb_values[index][step] = batchify(values[index],device).flatten()
                           rb_invalid_action_masks[step] = batchify(invalid_action_masks[index],device)
                    # compute episodic return
                    for i in range(len(indeces)):
                       index = indeces[i]
                       total_episodic_return[index] += rb_rewards[index][step].cpu().numpy()
 
                    # if we reach termination or truncation, end
                    if all([env.terminations[a] for a in env.terminations]) or all([env.terminations[a] for a in env.terminations]):
                       end_step = step
                       break

             # bootstrap value if not done
            with torch.no_grad():
                rb_advantages = {}
                rb_returns = {}
                for i in range(len(indeces)):
                   index = indeces[i]
                   rb_advantages[index] = torch.zeros_like(rb_rewards[index]).to(device)
                   for t in reversed(range(end_step)):
                       delta = (
                           rb_rewards[index][t]
                           + gamma * rb_values[index][t + 1] * rb_terms[index][t + 1]
                           - rb_values[index][t]
                       )
                       rb_advantages[index][t] = delta + gamma * gamma * rb_advantages[index][t + 1]
                   rb_returns[index] = rb_advantages[index] + rb_values[index]
            # convert our episodes to batch of individual transitions
            b_obs = {}
            b_logprobs = {}
            b_actions = {}
            b_returns = {}
            b_values = {}
            b_advantages = {}
            b_invalid_action_masks = {}
            for i in range(len(indeces)):
                index = indeces[i]
                b_obs[index] = torch.flatten(rb_obs[index][:end_step], start_dim=0, end_dim=1)
                b_logprobs[index] = torch.flatten(rb_logprobs[index][:end_step], start_dim=0, end_dim=1)
                b_actions[index] = torch.flatten(rb_actions[index][:end_step], start_dim=0, end_dim=1)
                b_returns[index] = torch.flatten(rb_returns[index][:end_step], start_dim=0, end_dim=1)
                b_values[index] = torch.flatten(rb_values[index][:end_step], start_dim=0, end_dim=1)
                b_advantages[index] = torch.flatten(rb_advantages[index][:end_step], start_dim=0, end_dim=1)
                b_invalid_action_masks[index] = torch.flatten(rb_invalid_action_masks[index][:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = {}
            batch_index = {}
            for i in range(len(indeces)):
               index = indeces[i]
               b_index[index] = np.arange(len(b_actions[index]))
               clip_fracs = []
               for repeat in range(3):
                 # shuffle the indices we use to access the data
                 np.random.shuffle(b_index[index])
                 for start in range(0, len(b_actions[index]), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index[index] = b_index[index][start:end]   
                    _, newlogprob, entropy, value = self.get_action_and_value(
                        b_obs[index][batch_index[index]], index,
                        action = b_actions[index].long()[batch_index[index]], 
                        invalid_action_masks = b_invalid_action_masks[index][batch_index[index]]
                    )
                    logratio = newlogprob - b_logprobs[index][batch_index[index]]
                    ratio = logratio.exp()

                    with torch.no_grad():
                       # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[index][batch_index[index]]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -b_advantages[index][batch_index[index]] * ratio
                    pg_loss2 = -b_advantages[index][batch_index[index]] * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   
                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[index][batch_index[index]]) ** 2
                    v_clipped = b_values[index][batch_index[index]] + torch.clamp(
                        value - b_values[index][batch_index[index]],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[index][batch_index[index]]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                    optimizer[index].zero_grad()
                    loss.backward()
                    optimizer[index].step()

               y_pred, y_true = b_values[index].cpu().numpy(), b_returns[index].cpu().numpy()
               var_y = np.var(y_true)
               explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
               
               policy_losses[index].append(pg_loss.item())
               value_function_loss[index].append(v_loss.item())
               explained_variance[index].append(explained_var.item())       

               print("**" + index + " Team**")
               print(f"Training episode {episode}")
               print(f"Episodic Return: {np.mean(total_episodic_return[index])}")
               print(f"Episode Length: {end_step}")
               print("")
               print(f"Value Loss: {v_loss.item()}")
               print(f"Policy Loss: {pg_loss.item()}")
               print(f"Old Approx KL: {old_approx_kl.item()}")
               print(f"Approx KL: {approx_kl.item()}")
               print(f"Clip Fraction: {np.mean(clip_fracs)}")
               print(f"Explained Variance: {explained_var.item()}")
               print("\n***************************************\n")
            print("\n-------------------------------------------\n") 
            # self.eval()
        
            if (episode + 1) % 100 == 0:
               # Save our model if it's time
               for i in range(len(self.NeuralNetworks)):
                   index = indeces[i]
                   network = self.NeuralNetworks[index]
                   torch.save(network.actor.state_dict(), PATH + '/ppo_actor_'+ network.strategy_type +'.pth')
                   torch.save(network.critic.state_dict(), PATH + '/ppo_critic_'+ network.strategy_type +'.pth')
                   
        
        # Plot training curves
        for i in range(len(indeces)):
            index = indeces[i]
            plt.figure(figsize=(12, 4))
            plt.grid()
            plt.subplot(1, 3, 1)
            plt.plot(explained_variance[index])
            plt.xlabel('Epoch')
            plt.ylabel('Explained Variance')
            plt.title('Explained Variance for ' + index + ' team')

            plt.subplot(1, 3, 2)
            plt.plot(policy_losses[index])
            plt.xlabel('Epoch')
            plt.ylabel('Policy Loss')
            plt.title('Policy Loss for ' + index + ' team')

            plt.subplot(1, 3, 3)
            plt.plot(value_function_loss[index])
            plt.xlabel('Epoch')
            plt.ylabel('Value Function Loss')
            plt.title('Value Function Loss for ' + index + ' team') 
            
            figure_file = PATH + "PPO_Analysis/PPO_Agent_Analysis_for_" + index + '_team.png'
            plt.savefig(figure_file) 
            
            plt.tight_layout()
            plt.show()

    def render_policy(self, PATH, env):
        """ RENDER THE POLICY """
        self.eval()
        PATH = PATH + "/Render_Policy/" 
        with torch.no_grad():
            # render 5 episodes out
            for episode in range(5):
                env.reset(seed=None)
                observation, reward, termination, truncation, info = env.last()
                obs = batchify_obs(observation, env.agent_selection, device)
                terms = [termination]
                truncs = [truncation]
                rewards = {}
                for agent in env.agents: 
                    rewards[agent] = []
                while not any(terms) and not any(truncs):
                  for agent in env.agents:
                     rewards[agent].append(reward)
                     action, logprob, _, value = self.get_action_and_value(obs, agent.get_player_strategy_name())
                     player_action = unbatchify(action)
                     env.step(player_action)
                     observation, reward, termination, truncation, info = env.last()
                     obs = batchify_obs(observation, agent, device)
                     terms.append(termination)
                     truncs.append(truncation)
                print("Episode: " + str(episode))
                # print(states)                           
                for agent in env.agents:
                    plt.figure()
                    x = [i+1 for i in range(len(rewards[agent]))]
                    plt.plot(x, rewards[agent])
                    plt.title('Running rewards of ' + agent.get_name() + '_Episode: ' + str(episode))
                    figure_file = PATH + agent.get_name() + '_Episode_' + str(episode) + '.png'     
                    plt.savefig(figure_file)
                for agent in env.agents:
                    print(agent.get_name())
                    print(agent.get_score())
                final_state = env.game_object.get_system_state()
                print("Episode: " + str(episode) + " ended in states: ")
                print(final_state)
                print("------------------------------------------------------------------")

def train(PATH, env, actor_model, critic_model, total_episodes):
    #print(f"Training", flush=True)
    model = create_ppo_agent(env, actor_model, critic_model)   
    # Train the PPO model with a specified total timesteps
    #print(model.device)
    model.learn(PATH, env, total_episodes = total_episodes)  
    # model.render_policy(PATH,env)

def create_ppo_agent(env, actor_model, critic_model):
      num_actions = env.action_space(env.possible_agents[0]).n
      observation_size = env.observation_space(env.possible_agents[0]).shape[0]
      # Create a model for PPO.
      model = Agent(env=env, in_dim=observation_size, out_dim=num_actions).to(device)
      # Tries to load in an existing actor/critic model to continue training on
      if actor_model != '' and critic_model != '':
         #print(f"Loading in {actor_model} and {critic_model}...", flush=True) 
         indeces = ["GOOD", "BAD"]
         for i in range(len(model.NeuralNetworks)):
             index = indeces[i]
             network = model.NeuralNetworks[index]
             network.actor.load_state_dict(torch.load(actor_model + network.strategy_type +'.pth'))
             print(actor_model + network.strategy_type +'.pth')
             network.critic.load_state_dict(torch.load(critic_model + network.strategy_type +'.pth'))
         print(f"Successfully loaded.", flush=True)
      elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
         print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
         sys.exit(0)
         return None
      else:
         print(f"Training from scratch.", flush=True)
      return model

if __name__ == "__main__":
    """ALGO PARAMS"""
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    total_episodes = 2

    """ ENV SETUP """
    env,_ = env_creator(model_file, AGENTS_TYPE[2], 0 )
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n
    observation_size = env.observation_space(env.possible_agents[0]).shape[0]
    max_cycles = env.game_object.get_max_steps() + 3
    stack_size = env.game_object.get_num_system_objects()
    depth = env.game_object.num_observation()
    #print(stack_size)
    """ LEARNER SETUP """
    # agent = Agent(num_actions=num_actions).to(device)
    #print(num_actions)
    ppo_agent = Agent(env=env, in_dim=observation_size, out_dim=num_actions).to(device)
    indeces = ["GOOD", "BAD"]
    optimizer = {}
    for i in range(len(indeces)):
        index = indeces[i]
        optimizer [index] = optim.Adam(ppo_agent.NeuralNetworks[index].parameters(), lr=0.001, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    
    rb_obs = {}
    rb_actions = {}
    rb_logprobs = {}
    rb_rewards = {}
    rb_terms = {}
    rb_values = {}
    rb_invalid_action_masks = {}

    for i in range(len(indeces)):
        index = indeces[i]
        rb_obs[index] = torch.zeros((max_cycles, num_agents, stack_size)).to(device)
        rb_actions[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device) 
        rb_logprobs[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)   
        rb_rewards[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)   
        rb_terms[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)    
        rb_values[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index))).to(device)
        agent = env.game_object.get_players_strategy_type(index)[0]
        #rb_invalid_action_masks[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index)) + (agent.get_num_valid_actions(),)).to(device)
        rb_invalid_action_masks[index] = torch.zeros((max_cycles, env.game_object.get_num_players_strategy_type(index)) + (depth + 1,)).to(device)
        #print(env.game_object.num_observation())

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            env.reset(seed=None) 
            # reset the episodic return
            total_episodic_return = {}
            for i in range(len(indeces)):
                index = indeces[i]
                total_episodic_return[index] = 0
            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                actions = {}
                logprobs= {}
                values  = {}
                rewards = {}
                terminations = {}
                invalid_action_masks = {}
                for i in range(len(indeces)):
                    index = indeces[i]
                    actions[index] = {}
                    logprobs[index] = {}
                    values[index] = {}
                    rewards[index] = {}
                    terminations[index] = {}
                    invalid_action_masks[index] = {}
                for agent in env.agents:
                    #print(env.agents)
                    observation, reward, termination, truncation, info = env.last()
                    #print(observation)
                    #print(".......................")
                    if termination or truncation:
                       action = None
                    else:
                       #print(observation)
                       invalid_action_masks[agent.get_player_strategy_name()][agent] = env.game_object.get_mask(agent)
                       obs = batchify_obs(observation, agent, device)
                       action_mask = {agent:env.game_object.get_mask(agent)}
                       action_mask = batchify(action_mask,device)
                       
                       #print(env.game_object.get_mask(agent))
                       # action_mask = batchify(invalid_action_masks[agent.get_player_strategy_name()][agent],device)

                       action, logprob, _, value = ppo_agent.get_action_and_value(obs, agent.get_player_strategy_name(), invalid_action_masks = action_mask)
                       action = unbatchify(action)[0]
                       logprob = unbatchify(logprob)
                       value = unbatchify(value)
                       actions[agent.get_player_strategy_name()][agent] = action
                       logprobs[agent.get_player_strategy_name()][agent] = logprob
                       values[agent.get_player_strategy_name()][agent] = value   
                    env.step(action)                        
                if (action != None):
                   for agent in env.agents:
                       #print(agent.get_player_strategy_name())
                       rewards[agent.get_player_strategy_name()][agent] = env.rewards[agent] 
                       terminations[agent.get_player_strategy_name()][agent] = env.terminations[agent] 
                   for i in range(len(indeces)):
                       index = indeces[i]
                       #print(env.observations) 
                       #print(step)
                       #print(rb_obs[index][step].size())
                       rb_obs[index][step] = batchify(env.observations, device) 
                       #print("///////////")
                       rb_rewards[index][step] = batchify(rewards[index], device)
                       rb_terms[index][step] = batchify(terminations[index], device)
                       rb_actions[index][step] = batchify(actions[index],device)
                       rb_logprobs[index][step] = batchify(logprobs[index],device)
                       rb_values[index][step] = batchify(values[index],device).flatten()
                       #print(invalid_action_masks[index])
                       rb_invalid_action_masks[index][step] = batchify(invalid_action_masks[index],device)

                   # compute episodic return
                   for i in range(len(indeces)):
                      index = indeces[i]
                      total_episodic_return[index] += rb_rewards[index][step].cpu().numpy()

                if all([env.terminations[a] for a in env.terminations]) or all([env.terminations[a] for a in env.terminations]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = {}
            rb_returns = {}
            for i in range(len(indeces)):
               index = indeces[i]
               rb_advantages[index] = torch.zeros_like(rb_rewards[index]).to(device)
               for t in reversed(range(end_step)):
                   delta = (
                       rb_rewards[index][t]
                       + gamma * rb_values[index][t + 1] * rb_terms[index][t + 1]
                       - rb_values[index][t]
                   )
                   rb_advantages[index][t] = delta + gamma * gamma * rb_advantages[index][t + 1]
               rb_returns[index] = rb_advantages[index] + rb_values[index]
        # convert our episodes to batch of individual transitions
        b_obs = {}
        b_logprobs = {}
        b_actions = {}
        b_returns = {}
        b_values = {}
        b_advantages = {}
        b_invalid_action_masks = {}
        for i in range(len(indeces)):
            index = indeces[i]
            b_obs[index] = torch.flatten(rb_obs[index][:end_step], start_dim=0, end_dim=1)
            b_logprobs[index] = torch.flatten(rb_logprobs[index][:end_step], start_dim=0, end_dim=1)
            b_actions[index] = torch.flatten(rb_actions[index][:end_step], start_dim=0, end_dim=1)
            b_returns[index] = torch.flatten(rb_returns[index][:end_step], start_dim=0, end_dim=1)
            b_values[index] = torch.flatten(rb_values[index][:end_step], start_dim=0, end_dim=1)
            b_advantages[index] = torch.flatten(rb_advantages[index][:end_step], start_dim=0, end_dim=1)
            b_invalid_action_masks[index] = torch.flatten(rb_invalid_action_masks[index][:end_step], start_dim=0, end_dim=1)
       
        # Optimizing the policy and value network
        b_index = {}
        batch_index = {}
        for i in range(len(indeces)):
           index = indeces[i]
           b_index[index] = np.arange(len(b_actions[index]))
           clip_fracs = []
           for repeat in range(3):
               # shuffle the indices we use to access the data
               np.random.shuffle(b_index[index])
               for start in range(0, len(b_actions[index]), batch_size):
                   # select the indices we want to train on
                   end = start + batch_size
                   batch_index[index] = b_index[index][start:end]   
                   _, newlogprob, entropy, value = ppo_agent.get_action_and_value(
                       b_obs[index][batch_index[index]], index,
                       action = b_actions[index].long()[batch_index[index]], 
                       invalid_action_masks = b_invalid_action_masks[index][batch_index[index]]
                   )
                   logratio = newlogprob - b_logprobs[index][batch_index[index]]
                   ratio = logratio.exp()
                   
                   with torch.no_grad():
                      # calculate approx_kl http://joschu.net/blog/kl-approx.html
                       old_approx_kl = (-logratio).mean()
                       approx_kl = ((ratio - 1) - logratio).mean()
                       clip_fracs += [
                           ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                       ]

                   # normalize advantaegs
                   advantages = b_advantages[index][batch_index[index]]
                   advantages = (advantages - advantages.mean()) / (
                       advantages.std() + 1e-8
                   )

                   # Policy loss
                   pg_loss1 = -b_advantages[index][batch_index[index]] * ratio
                   pg_loss2 = -b_advantages[index][batch_index[index]] * torch.clamp(
                       ratio, 1 - clip_coef, 1 + clip_coef
                   )
                   pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   
                   # Value loss
                   value = value.flatten()
                   v_loss_unclipped = (value - b_returns[index][batch_index[index]]) ** 2
                   v_clipped = b_values[index][batch_index[index]] + torch.clamp(
                       value - b_values[index][batch_index[index]],
                       -clip_coef,
                       clip_coef,
                   )
                   v_loss_clipped = (v_clipped - b_returns[index][batch_index[index]]) ** 2
                   v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                   v_loss = 0.5 * v_loss_max.mean()

                   entropy_loss = entropy.mean()
                   loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                   optimizer[index].zero_grad()
                   loss.backward()
                   optimizer[index].step()

           y_pred, y_true = b_values[index].cpu().numpy(), b_returns[index].cpu().numpy()
           var_y = np.var(y_true)
           explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                     
           print("**" + index + " Team**")
           print(f"Training episode {episode}")
           print(f"Episodic Return: {np.mean(total_episodic_return[index])}")
           print(f"Episode Length: {end_step}")
           print("")
           print(f"Value Loss: {v_loss.item()}")
           print(f"Policy Loss: {pg_loss.item()}")
           print(f"Old Approx KL: {old_approx_kl.item()}")
           print(f"Approx KL: {approx_kl.item()}")
           print(f"Clip Fraction: {np.mean(clip_fracs)}")
           print(f"Explained Variance: {explained_var.item()}")
           print("\n***************************************\n")
        print("\n-------------------------------------------\n")
    
    PATH = './PPO_Agent'
    # Save our model if it's time
    for i in range(len(indeces)):
        index = indeces[i]
        torch.save(ppo_agent.NeuralNetworks[index].actor.state_dict(), PATH + '/ppo_actor_' + ppo_agent.NeuralNetworks[index].strategy_type +'.pth')
        torch.save(ppo_agent.NeuralNetworks[index].critic.state_dict(), PATH + '/ppo_critic_' + ppo_agent.NeuralNetworks[index].strategy_type +'.pth')
    
    print(f"Device: {device}")
    
    """ RENDER THE POLICY """
    env,_ = env_creator(model_file, AGENTS_TYPE[2], 2 if model_file == 'model.xml' else 0)
    ppo_agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            env.reset(seed=None)
            observation, reward, termination, truncation, info = env.last()
            obs = batchify_obs(observation, env.agent_selection, device)
            terms = [termination]
            truncs = [truncation]
            i=0
            while not any(terms) and not any(truncs):
              terms = [False]
              truncs = [False]
              for agent in env.agents: 
                 action, logprob, _, value = ppo_agent.get_action_and_value(obs, agent.get_player_strategy_name())
                 player_action = unbatchify(action)[0]
                 env.step(player_action)
                 observation, reward, termination, truncation, info = env.last()
                 obs = batchify_obs(observation, agent, device)
                 terms.append(termination)
                 truncs.append(truncation)
                 i=i+1
    
    for player in env.game_object.get_players():
       print(player.get_name())
       print(player.get_score())
    print("Done")
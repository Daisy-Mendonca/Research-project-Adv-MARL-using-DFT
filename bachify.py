import numpy as np
import torch


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
    #print(x)
    x = np.stack([x[a] for a in x], axis=0)
    #print(".................")
    #print(x)
    # convert to torch
    x = torch.tensor(x).to(device)
    #print(x)
    #print(x.size())
    return x

def unbatchify(x):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    # x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x[0]  
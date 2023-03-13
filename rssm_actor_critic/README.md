## PyTorch Implementation of World Model with Actor-Critic for Reinforcement Learning

This is a PyTorch implementation of the World Model with Actor-Critic architecture for reinforcement learning. The World Model consists of a Recurrent State-Space Model (RSSM) for modeling the environment, an actor network for selecting actions, and a critic network for estimating the state value function.

The RSSM is implemented as a PyTorch module with three subnetworks: an encoder network, a transition network, and a decoder network. The encoder network takes the current observation and action as input and outputs a latent state. The transition network takes the latent state and action as input and outputs the next latent state. The decoder network takes the current latent state, observation, and action as input and outputs the predicted next observation. The RSSM also includes a prior distribution over the latent state.

The actor network is implemented as a PyTorch module that takes the current observation as input and outputs a probability distribution over the possible actions.

The critic network is implemented as a PyTorch module that takes the current observation and action as input and outputs an estimate of the state value function.

The train function uses the Proximal Policy Optimization (PPO) algorithm to update the actor and critic networks. During each episode, the actor network selects actions based on the current observation and the critic network estimates the state value function. The RSSM is updated using the current observation, action, and predicted next observation. The actor and critic networks are updated using the PPO algorithm.

### Requirements
This implementation requires the following Python packages:
* PyTorch
* Gymnasium
* Matplotlib
* NumPy

### Usage
To use this implementation, you can create an instance of the RSSM class, the Actor class, and the Critic class. Then you can call the train function to train the model on an OpenAI Gym environment.
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import numpy as np

import pdb

class RSSM(nn.Module):
    ...

class Actor(nn.Module):
    ...

class Critic(nn.Module):
    ...

def train(num_episodes, env, rssm, actor, critic, optimizer_actor, optimizer_critic, gamma, device):
    ...
```
The train function takes the following arguments:
* num_episodes: The number of episodes to train for.
* env: The OpenAI Gym environment to train on.
* rssm: An instance of the RSSM class.
* actor: An instance of the Actor class.
* critic: An instance of the Critic class.
* optimizer_actor: The optimizer to use for training the actor.
* optimizer_critic: The optimizer to use for training the critic.
* gamma: The discount factor.
* device: The device to use for training (e.g. 'cpu' or 'cuda').

### Example
```python
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

rssm = RSSM(obs_dim=obs_dim, act_dim=act_dim, state_dim=32, hidden_dim=64)
actor = Actor(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=32)
critic = Critic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=32)

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=1e-4)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

actor_losses, critic_losses, rewards = train(num_episodes=100, env=env, rssm=rssm, actor=actor, critic=critic, optimizer_actor=optimizer_actor, optimizer_critic=optimizer_critic, gamma=0.99, device=device)
```
This will train the agent for 100 episodes and return the lists of actor losses, critic losses, and rewards for each episode.

## Acknowledgements

This implementation was inspired by the paper ["Dream to Control: Learning Behaviors by Latent Imagination"](https://arxiv.org/abs/1912.01603) by Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson.

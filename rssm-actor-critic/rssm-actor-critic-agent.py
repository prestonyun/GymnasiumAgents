import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import pdb

class RSSM(nn.Module):
    """A Recurrent State-Space Model (RSSM) implementation in PyTorch.
    
    Args:
        obs_dim (int): The dimension of observation space.
        act_dim (int): The dimension of action space.
        state_dim (int): The dimension of latent state space.
        hidden_dim (int): The dimension of hidden state space.
    """
    def __init__(self, obs_dim, act_dim, state_dim, hidden_dim):
        super(RSSM, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + act_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        # Define the transition network
        self.transition = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, obs_dim),
        )

        # Define the prior state distribution
        self.prior = torch.distributions.Normal(torch.zeros(state_dim).to('cuda'), torch.ones(state_dim).to('cuda'))

    def forward(self, obs, action, state):
        """Feedforward function of the RSSM.

        Args:
            obs (torch.Tensor): A tensor of shape (batch_size, obs_dim) representing the observation.
            action (torch.Tensor): A tensor of shape (batch_size, act_dim) representing the action.
            state (torch.Tensor): A tensor of shape (batch_size, state_dim) representing the latent state.

        Returns:
            obs_pred (torch.Tensor): A tensor of shape (batch_size, obs_dim) representing the predicted observation.
            next_state (torch.Tensor): A tensor of shape (batch_size, state_dim) representing the predicted next state.
            kl_div (torch.Tensor): A tensor of shape (batch_size,) representing the KL divergence between the prior and current state distribution.
            likelihood (torch.distributions.Normal): A Normal distribution object representing the likelihood of the observation given the current state.
        """
        # Encode the observation and action into the current state
        action = torch.unsqueeze(action, dim=-1).to('cuda')
        x = torch.cat([obs, action], dim=-1).to('cuda')
        h = self.encoder(x)

        # Calculate the next state using the current state and action
        x = torch.cat([h, action], dim=-1).to('cuda')
        next_state = self.transition(x)

        # Calculate the observation from the current state
        y = torch.cat([h, obs, action], dim=-1).to('cuda')
        obs_pred = self.decoder(y)

        # Calculate the likelihood of the observation given the current state
        likelihood = torch.distributions.Normal(obs_pred, 0.1)

        # Calculate the KL divergence between the prior and current state distribution
        kl_div = torch.distributions.kl.kl_divergence(torch.distributions.Normal(state.to('cuda'), torch.ones(state.shape).to('cuda')), self.prior)

        # Return the predicted observation, next state, and KL divergence
        return obs_pred, next_state, kl_div, likelihood

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        """
        Actor network for a discrete action space.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden_dim (int): Dimension of the hidden layer.
            num_actions (int): Number of discrete actions.
        """
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.num_actions = act_dim + 1

        # Define the actor network
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.num_actions),
            nn.ReLU(),
        )

        self.fc = nn.Linear(self.num_actions, hidden_dim)

        self.logits = nn.Linear(hidden_dim, self.num_actions)

    def forward(self, obs):
        """
        Computes the forward pass of the Actor network.

        Args:
            obs (torch.Tensor): Current observation, shape (batch_size, obs_dim).

        Returns:
            action_probs (torch.Tensor): Probability distribution over discrete actions, shape (batch_size, num_actions).
        """
        # Encode the observation into a hidden representation
        obs_h = self.obs_encoder(obs)
        x = F.relu(self.fc(obs_h))

        # Calculate the logits for the discrete action space
        logits = self.logits(x)

        # Apply softmax to the logits to get the probability distribution over the discrete actions
        action_probs = F.softmax(logits, dim=-1)

        return action_probs


class Critic(nn.Module):
    """A critic implementation in PyTorch.

    Args:
        obs_dim (int): The dimension of observation space.
        act_dim (int): The dimension of action space.
        hidden_dim (int): The dimension of hidden state space.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(Critic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        # Define the critic network
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),)

        self.fc = nn.Linear(hidden_dim + act_dim, hidden_dim)

        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        """Feedforward function of the critic.

        Args:
            obs (torch.Tensor): A tensor of shape (batch_size, obs_dim) representing the observation.
            action (torch.Tensor): A tensor of shape (batch_size, act_dim) representing the action.

        Returns:
            value (torch.Tensor): A tensor of shape (batch_size, 1) representing the state value.
        """
        # Encode the observation and action into a hidden representation
        obs_h = self.obs_encoder(obs)
        action = torch.unsqueeze(action, dim=-1)
        x = torch.cat([obs_h, action], dim=-1)
        x = self.fc(x)
        x = F.relu(x, inplace=False)

        #pdb.set_trace()
        # Calculate the state value from the hidden representation
        v = self.value(x)
        return v

def train(num_episodes, env, rssm, actor, critic, optimizer_actor, optimizer_critic, gamma, device):
    """Train function for the PPO algorithm.

    Args:
        num_episodes (int): The number of episodes to train for.
        env (gym.Env): The environment to train on.
        rssm (RSSM): The world model to use for planning.
        actor (Actor): The actor network to optimize.
        critic (Critic): The critic network to optimize.
        optimizer_actor (torch.optim.Optimizer): The optimizer to use for training the actor.
        optimizer_critic (torch.optim.Optimizer): The optimizer to use for training the critic.
        gamma (float): The discount factor.

    Returns:
        actor_losses (list): A list of the actor losses for each episode.
        critic_losses (list): A list of the critic losses for each episode.
        rewards (list): A list of the total rewards for each episode.
    """
    actor_losses, critic_losses, rewards = [], [], []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            env.render()
            # Sample an action from the Actor network
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action_probs = actor(obs)
            action = torch.multinomial(action_probs, 1).item()

            # Take a step in the environment with the selected action
            next_obs, reward, done, _, info = env.step(action)

            # Update the RSSM world model
            obs_tensor = obs.clone().to(device)
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0).to(device)

            state_tensor = torch.zeros(rssm.state_dim).to(device)
            obs_pred, next_state, kl_div, likelihood = rssm(obs_tensor, action_tensor, state_tensor)

            # Update the Critic network

            value = critic(obs_tensor, action_tensor)

            # Calculate the TD error
            next_action_probs = actor(next_obs_tensor)
            next_action = torch.multinomial(next_action_probs, 1).item()
            next_action_tensor = torch.tensor(next_action, dtype=torch.float32).unsqueeze(0).to(device)
            next_value = critic(next_obs_tensor, next_action_tensor) if not done else 0
            td_error = reward + gamma * next_value - value

            # Update the Actor network
            log_prob = torch.log(action_probs[0, action])
            actor_loss = -(log_prob * td_error.detach()).mean()

            # Update the Critic network
            critic_loss = td_error.pow(2).mean()

            optimizer_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            obs = next_obs
            episode_rewards += reward

            if episode_rewards > 500:
                break

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        rewards.append(episode_rewards)
        print(f"Episode {i+1} - Total Reward: {episode_rewards:.2f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")

    return actor_losses, critic_losses, rewards


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    env = gym.make('CartPole-v1', render_mode='human')#, render_mode='human')

    num_episodes = 200
    lr = 1e-3
    gamma = 0.99
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n - 1
    state_dim = 5
    hidden_dim = 128

    rssm = RSSM(obs_dim, act_dim, state_dim, hidden_dim).to(device)
    actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
    critic = Critic(obs_dim, act_dim, hidden_dim).to(device)
    for param in actor.parameters():
        param = param.to(device)
    for param in critic.parameters():
        param = param.to(device)
    optimizer_actor = torch.optim.Adam(list(actor.parameters()), lr=lr)
    optimizer_critic = torch.optim.Adam(list(critic.parameters()), lr=lr)

    actor_losses, critic_losses, rewards = train(num_episodes, env, rssm, actor, critic, optimizer_actor, optimizer_critic, gamma, device)

    # Plot the actor and critic losses
    fig, ax = plt.subplots()
    ax.plot(actor_losses, label='Actor Loss')
    ax.plot(critic_losses, label='Critic Loss')
    ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    plt.show()

    # Plot the rewards
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    plt.show()

    obs, _ = env.reset()
    done = False
    while not done:
        # Sample a discrete action from the probability distribution output by the Actor network
        action_probs = actor(torch.tensor(obs, dtype=torch.float32).to(device))
        action = torch.multinomial(action_probs, 1).item()

        obs, reward, done, _, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main()
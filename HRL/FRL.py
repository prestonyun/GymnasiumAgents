import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the high-level manager
class HighLevelManager(nn.Module):
    def __init__(self, obs_shape, subgoal_shape):
        super(HighLevelManager, self).__init__()
        self.fc1 = nn.Linear(obs_shape[0], 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, subgoal_shape[0]).to(device)
        self.temperature = 1.0

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        subgoal = F.gumbel_softmax(logits/self.temperature + gumbel_noise, tau=1.0, dim=-1)
        subgoal = subgoal * 2 - 1
        return subgoal

class LowLevelWorker(nn.Module):
    def __init__(self, obs_shape, action_shape, noise_std=0.1):
        super(LowLevelWorker, self).__init__()
        self.fc1 = nn.Linear(obs_shape, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, 1).to(device)
        self.noise_std = noise_std
        
    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        action_mean = torch.sigmoid(self.fc3(x))
        action_noise = torch.randn_like(action_mean) * self.noise_std
        action = torch.clamp(action_mean + action_noise, 0.0, 1.0) * 2.0 - 1.0
        return action


# Define the SAC algorithm
class SAC:
    def __init__(self, env, subgoal_shape, worker_models, worker_lr=1e-4, critic_lr=1e-3, actor_lr=1e-3, gamma=0.99, alpha=0.2, alpha_lr=1e-4, min_alpha=0.01, tau=0.005, replay_buffer_size=int(1e6), batch_size=128):
        self.env = env
        self.subgoal_shape = subgoal_shape
        self.worker_models = worker_models
        self.worker_lr = worker_lr
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_lr = alpha_lr
        self.min_alpha = min_alpha
        self.tau = tau
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size

        # Define the actor, critic, and entropy optimizer for the high-level manager
        self.high_level_actor = HighLevelManager(env.observation_space.shape, subgoal_shape).to(device)
        self.high_level_critic1 = nn.Linear(env.observation_space.shape[0] + subgoal_shape[0], 64).to(device)
        self.high_level_critic2 = nn.Linear(64, 1).to(device)
        self.high_level_target_critic1 = nn.Linear(env.observation_space.shape[0] + subgoal_shape[0], 64).to(device)
        self.high_level_target_critic2 = nn.Linear(64, 1).to(device)
        self.high_level_actor_optimizer = optim.Adam(self.high_level_actor.parameters(), lr=actor_lr)
        self.high_level_critic_optimizer = optim.Adam(list(self.high_level_critic1.parameters()) + list(self.high_level_critic2.parameters()), lr=critic_lr)
        self.high_level_entropy_optimizer = optim.Adam(self.high_level_actor.parameters(), lr=actor_lr)

        # Define the critic and entropy optimizer for the low-level workers
        self.worker_critic1s = []
        self.worker_critic2s = []
        self.worker_target_critic1s = []
        self.worker_target_critic2s = []
        self.worker_entropy_optimizers = []
        self.worker_critic_optimizers = []
        for i in range(len(worker_models)):
            worker_critic1 = nn.Linear(env.observation_space.shape[0] + subgoal_shape[0] + 1, 64)
            worker_critic2 = nn.Linear(64, 1)
            worker_target_critic1 = nn.Linear(env.observation_space.shape[0] + subgoal_shape[0] + 1, 64)
            worker_target_critic2 = nn.Linear(64, 1)
            self.worker_critic_optimizer = optim.Adam(list(worker_critic1.parameters()) + list(worker_critic2.parameters()), lr=critic_lr)
            worker_entropy_optimizer = optim.Adam(worker_models[i].parameters(), lr=worker_lr)
            self.worker_critic1s.append(worker_critic1.to(device))
            self.worker_critic2s.append(worker_critic2.to(device))
            self.worker_target_critic1s.append(worker_target_critic1.to(device))
            self.worker_target_critic2s.append(worker_target_critic2.to(device))
            self.worker_entropy_optimizers.append(worker_entropy_optimizer)
            self.worker_critic_optimizers.append(self.worker_critic_optimizer)

        # Initialize the replay buffer
        self.replay_buffer = []
        self.replay_buffer_index = 0

    # Compute the Q-value for the high-level manager
    def compute_high_level_q(self, obs, subgoal):
        x = torch.cat([obs, subgoal], dim=-1)
        x = torch.relu(self.high_level_critic1(x))
        q_value = self.high_level_critic2(x)
        return q_value.to(device)

    # Compute the Q-value for the low-level worker
    def compute_worker_q(self, obs, subgoal, action, i):
        x = torch.cat([obs, subgoal], dim=-1)
        #action = torch.FloatTensor(action)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)
        x = torch.cat([x, action], dim=-1)
        x = torch.relu(self.worker_critic1s[i](x))
        q_value = self.worker_critic2s[i](x)
        return q_value.to(device)

    # Update the high-level manager
    def update_high_level(self):
        # Sample a batch of transitions from the replay buffer
        indices = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        obs_batch, subgoal_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*[self.replay_buffer[i] for i in indices])
        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
        subgoal_batch = torch.FloatTensor(np.array(subgoal_batch)).to(device)
        action_batch = torch.FloatTensor(np.array(action_batch)).to(device)
        reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(device)
        done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

        # Compute the target Q-value for the high-level manager
        with torch.no_grad():
            next_subgoal_batch = self.high_level_actor(next_obs_batch)
            next_q_value = self.compute_high_level_q(next_obs_batch, next_subgoal_batch)
            target_q_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * next_q_value

        # Compute the Q-value for the high-level manager
        q_value = self.compute_high_level_q(obs_batch, subgoal_batch)

        # Compute the actor and critic losses
        actor_loss = -torch.mean(self.alpha * self.high_level_actor(obs_batch) - self.compute_high_level_q(obs_batch, self.high_level_actor(obs_batch)))
        critic_loss = F.mse_loss(q_value, target_q_value)

        # Update the actor and critic parameters
        self.high_level_actor_optimizer.zero_grad()
        actor_loss.backward()
        self.high_level_actor_optimizer.step()

        self.high_level_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.high_level_critic_optimizer.step()

        # Update the entropy parameter
        alpha_loss = -torch.mean(self.alpha * (torch.log(torch.tensor(self.alpha)) - torch.log(self.high_level_actor(obs_batch))))
        self.high_level_entropy_optimizer.zero_grad()
        alpha_loss.backward()
        self.high_level_entropy_optimizer.step()

    def update_workers(self):
        # Update the low-level workers
        for i in range(len(self.worker_models)):
            # Sample a batch of transitions from the replay buffer
            indices = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
            obs_batch, subgoal_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*[self.replay_buffer[i] for i in indices])
            obs_batch = torch.FloatTensor(np.array(obs_batch)).to(device)
            subgoal_batch = torch.FloatTensor(np.array(subgoal_batch)).to(device)
            action_batch = torch.FloatTensor(np.array(action_batch)).to(device)
            reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
            next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(device)
            done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

            # Compute the target Q-value for the low-level worker
            with torch.no_grad():
                next_action_batch = self.worker_models[i](next_obs_batch)
                next_subgoal_batch = subgoal_batch.clone()
                next_subgoal_batch[:,i] = next_action_batch.squeeze()
                next_q_value = self.compute_worker_q(next_obs_batch, next_subgoal_batch, next_action_batch, i)
                target_q_value = reward_batch.unsqueeze(-1) + (1 - done_batch.unsqueeze(-1)) * self.gamma * next_q_value

            # Compute the Q-value for the low-level worker
            q_value = self.compute_worker_q(obs_batch, subgoal_batch, action_batch[:, i:i+1], i)

            # Compute the actor and critic losses
            actor_loss = -torch.mean(self.alpha * self.high_level_actor(obs_batch) - self.compute_worker_q(obs_batch, subgoal_batch, self.high_level_actor(obs_batch)[:, i], i))
            critic_loss = F.mse_loss(q_value, target_q_value)

            # Update the actor and critic parameters
            self.worker_entropy_optimizers[i].zero_grad()
            actor_loss.backward()
            self.worker_entropy_optimizers[i].step()

            self.worker_critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.worker_critic_optimizer.step()


    def update_alpha(self, step):
        self.alpha = max(self.alpha * (1-step / 100000), 0.01)

    # Train the SAC algorithm
    def train(self, num_episodes):
        obs, _ = self.env.reset()
        for episode in range(num_episodes):
            #env.render()
            subgoal = self.high_level_actor(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
            done = False
            total_reward = 0
            episode_steps = 0
            action = np.zeros((len(self.worker_models), 1))
            while not done and episode_steps < self.env._max_episode_steps:
                # Choose an action from the low-level worker
                for i in range(len(self.worker_models)):
                    action[i] = self.worker_models[i](torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
                action = action.squeeze()
                # Execute the action and observe the next state and reward
                next_obs, reward, done, _, _ = self.env.step(action)
                # Store the transition in the replay buffer
                self.replay_buffer.append((obs, subgoal, action, reward, next_obs, done))
                while len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer.pop(0)

                # Update the high-level manager and low-level workers
                if len(self.replay_buffer) > self.batch_size:
                    self.update_high_level()
                    self.update_workers()

                obs = next_obs
                subgoal = self.high_level_actor(torch.FloatTensor(obs).to(device)).detach().cpu().numpy()
                total_reward += reward
                episode_steps += 1

            print("Episode {} - Total reward: {}, alpha: {}".format(episode, total_reward, self.alpha))
            #print(subgoal)
            obs, _ = self.env.reset()

# Test the SAC algorithm
env = gym.make('BipedalWalker-v3')#, render_mode='human')
subgoal_shape = (4,)
worker_models = [LowLevelWorker(env.observation_space.shape[0], env.action_space.shape[0]) for _ in range(subgoal_shape[0])]

sac = SAC(env, subgoal_shape, worker_models)

sac.train(100)
torch.save(sac.high_level_actor.state_dict(), 'high_level_actor.pth')
for i, model in enumerate(sac.worker_models):
    torch.save(model.state_dict(), 'worker_model_{}.pth'.format(i))
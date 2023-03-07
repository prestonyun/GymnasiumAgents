# Reinforcement Learning Agents for OpenAI Gym Environments
This repository contains multiple Reinforcement Learning agents that solve different OpenAI Gym environments.

## Installation
To install the required packages, run:
```
pip install -r requirements.txt
```

## Usage
To use the agents, simply navigate to the environment you want to solve, and run the corresponding script. For example, to run the Q-learning agent for the CartPole environment, navigate to the prioritized-dqn-agent directory and run:
```
python prioritized-dqn-agent.py
```

## Available agents
Currently, the following agents are available:
* DQN with Prioritized Replay
* RSSM with Actor-Critic
More agents will be added in the future.

## Environments
The agents were tested on the following environments:
* CartPole-v1

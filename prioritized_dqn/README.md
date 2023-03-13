# Prioritized DQN Agent

This is an implementation of a Prioritized DQN agent for the CartPole-v1 environment in the OpenAI fork Gymnasium. The agent uses prioritized experience replay to learn to balance a pole on a cart by moving the cart left or right. 

## Dependencies

This project requires the following dependencies:
- Python 3.x
- Gymnasium
- NumPy
- PyTorch
- Matplotlib

## Usage

To run the agent on the `CartPole-v1` environment, simply execute the `main` function in `prioritized_dqn.py`. You can adjust the hyperparameters of the agent in the `__init__` function of the `PrioritizedDQNAgent` class.

```python
def main():

    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PrioritizedDQNAgent(obs_dim, n_actions, 16, 1e-3, 0.99, 100000, 8, 0.6, 0.4, 10000000)
    rewards, epsilons = agent.learn(500, 1000, 1.0, 0.01, 0.9955, env)

    plot_learning_curve(rewards, epsilons)

if __name__ == "__main__":
    main()
```
    
## References
1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. arXiv preprint arXiv:1511.05952.
2. OpenAI, Farama Foundation. CartPole-v1. https://gymnasium.farama.org

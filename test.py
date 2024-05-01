import gym
import gym_sumo
import numpy as np
import torch
import random
import torch.nn as nn
from collections import namedtuple
import gym_sumo.envs.sumo_env as sumo_env

# Define the Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1404, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Update output_dim here
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the DQN Agent
class DQNAgent:
    def __init__(self, input_dim, output_dim, model_path, gamma, epsilon, target_update):
        self.policy_net = QNetwork(input_dim, output_dim)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
        self.steps_done = 0
        self.batch_size = 128
        self.gamma = gamma
        self.epsilon = epsilon
    
    # def select_action(self, state):
    #     with torch.no_grad():
    #         return self.policy_net(state).max(1)[1].view(1, 1)
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return torch.tensor([[random.randint(0, 1)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
    

    
# Load the trained model
model_path = 'C:\\Users\\Sande\\Desktop\\Safe-Intersection-Navigation-AVs-using-Reinforcement-Learning\\models\\model_episode_200.pth'

# Set up the environment
env_name = 'sumo-v0'  # Update with your environment name
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim, gamma=0.99, epsilon=0.1, target_update=10, model_path=model_path)


# Testing loop
num_episodes = 10  # Update with the number of episodes you want to test
for episode in range(num_episodes):
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action.item())
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not done else None
        state = next_state
        # env.render()  # Uncomment if you want to render the environment during testing
    print('Episode {}: Total Reward = {}'.format(episode, total_reward))

# Close environment
env.close()
    
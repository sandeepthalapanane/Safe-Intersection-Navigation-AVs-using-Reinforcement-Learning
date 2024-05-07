import gym
import gym_sumo
import numpy as np
import torch
import random
import torch.nn as nn
from collections import namedtuple
import gym_sumo.envs.sumo_env as sumo_env
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='test_log.txt', level=logging.INFO, format='%(message)s')


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(1404, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim) 
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, input_dim, output_dim, model_path, gamma, epsilon, target_update):
        self.policy_net = QNetwork(input_dim, output_dim)
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()
        self.steps_done = 0
        self.batch_size = 128
        self.gamma = gamma
        self.epsilon = epsilon
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return torch.tensor([[random.randint(0, 1)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
    

    

model_path = 'C:\\Users\\Sande\\Desktop\\Safe-Intersection-Navigation-AVs-using-Reinforcement-Learning\\models\\1200_episode_model.pth'


env_name = 'sumo-v0' 
env_args = {'mode': 'test'}
env = gym.make(env_name, **env_args)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim, gamma=0.99, epsilon=0.1, target_update=10, model_path=model_path)
num_episodes = 201
start_episode = 1
scenarios = 2
sceanrio_update = (num_episodes - start_episode)/scenarios
re_wards = []

for episode in range(1, num_episodes, 1):
    if (episode % 101 == 0) and (episode > 0):
        env.scenario_counter(int(episode/sceanrio_update))
        print("Testing the next scenario")
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state_tensor)
        next_state, reward, done, info = env.step(action.item())
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not done else None
        state = next_state
        # env.render()
    
    re_wards.append(total_reward)
    terminalType = info["information"]
    print('Episode {}: Total Reward = {}, Result: {}'.format(episode, round(total_reward,3), terminalType))
    logging.info('Episode {}: Total Reward = {}, Result: {}'.format(episode, round(total_reward,3), terminalType))

file_path = 'test_log.txt'
with open(file_path, 'r') as file:
    data = file.readlines()

# Parse the data to extract episode numbers and total rewards
episodes = []
rewards = []
for line in data:
    parts = line.strip().split(":")
    if len(parts) >= 2:
        episode = int(parts[0].split()[1])
        reward_str = parts[1].split("=")[1].split(",")[0].strip()
        try:
            reward = float(reward_str)
            episodes.append(episode)
            rewards.append(reward)
        except ValueError:
            print(f"Skipping line: {line}")

# Plot the total rewards
plt.plot(episodes, rewards, linestyle='-')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show()

env.close()
    

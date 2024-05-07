import gym
import gym_sumo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple
import os
import gym_sumo.envs.sumo_env as sumo_env
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(message)s')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class Replay_Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


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
    def __init__(self, input_dim, output_dim, lr, gamma, epsilon, target_update):
        self.policy_net = QNetwork(input_dim, output_dim)
        self.target_net = QNetwork(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = Replay_Memory(10000)
        self.batch_size = 128
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.steps_done = 0
        self.output_dim = output_dim
    
    def select_action(self, state):
        self.steps_done += 1
        if random.random() < self.epsilon:
            return torch.tensor([[random.randint(0, 1)]], dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) if len(s.shape) == 3 else torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])
        state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) if len(s.shape) == 3 else torch.tensor(s, dtype=torch.float32) for s in batch.state])
        action_batch = torch.cat([torch.tensor(a, dtype=torch.long) for a in batch.action])
        reward_batch = torch.cat([torch.tensor(r, dtype=torch.float32) for r in batch.reward])
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
env_name = 'sumo-v0'
env_args = {'mode': 'train'}
env = gym.make(env_name, **env_args)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
model_dir = "models/"


agent = DQNAgent(input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=0.1, target_update=5)


if os.path.exists(model_dir):
    saved_models = os.listdir(model_dir)
    if saved_models:
        saved_models.sort()
        last_model = saved_models[-1]
        last_episode = int(last_model.split("_")[0])
        start_episode = last_episode + 1
        model_path = os.path.join(model_dir, last_model)
        agent.policy_net.load_state_dict(torch.load(model_path))
        env.scenario_counter(int(last_episode/100))
        print("Loaded model weights from", model_path)
    else:
        start_episode = 0
else:
    start_episode = 0

num_episodes = 1200
scenarios = 12
sceanrio_update = num_episodes - start_episode/scenarios

re_wards = []

for episode in range(start_episode, num_episodes, 1):
    if (episode % 100 == 0) and (episode > 0):
        env.scenario_counter(int(episode/100))
        formatted_episode = str(episode).zfill(4)
        model_path = model_dir + "{}_episode_model.pth".format(formatted_episode)
        torch.save(agent.policy_net.state_dict(), model_path)
        print("Model saved at episode", episode)
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0) 
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state_tensor)
        next_state, reward, done, info = env.step(action.item())
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32, ).unsqueeze(0) if not done else None
        agent.memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([done], dtype=torch.bool))
        state = next_state
        agent.optimize_model()
        # env.render() 
    if episode % agent.target_update == 0:
        agent.update_target_network()

    re_wards.append(total_reward)
    terminalType = info["information"]
    print('Episode {}: Total Reward = {}, Result: {}'.format(episode, round(total_reward,3), terminalType))
    logging.info('Episode {}: Total Reward = {}, Result: {}'.format(episode, round(total_reward,3), terminalType))
    


torch.save(agent.policy_net.state_dict(), 'models/1200_episode_model.pth')


file_path = 'training_log.txt'
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

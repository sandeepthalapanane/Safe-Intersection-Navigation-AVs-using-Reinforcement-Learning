# import gym

# import gym_sumo
# import numpy as np

# game_name = 'sumo-v0'
# print("Starting", game_name)

# env = gym.make(game_name)
# print("observation space", env.observation_space)
# print("action space", env.action_space) 

# frameskip = 1
# ### SETUP SCENARIO ###
# for i in range(1000):

# 	env.reset()
# 	done = False
# 	cumulative_reward = 0
# 	# for j in xrange(100):
# 	while not done:

# 		action = np.random.randint(0,2)
		
# 		for f in range(frameskip):
# 			observation, reward, done, info = env.step(action)
# 			# print info
# 			cumulative_reward += reward
		
# 		env.render()
		
# 		if done==True:
# 			break
# 	print('reward',cumulative_reward)
import gym
import gym_sumo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple

# Define the Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
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

# Define the Q-Network
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
        x = torch.sigmoid(x) 
        x = torch.round(x) 
        return x

# Define the DQN Agent
class DQNAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, epsilon, target_update):
        self.policy_net = QNetwork(input_dim, output_dim)
        self.target_net = QNetwork(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
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
        loss = nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
# Set up the environment
env_name = 'sumo-v0'
env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
print("input_dim", input_dim)
output_dim = env.action_space.n

# Initialize the DQN agent
agent = DQNAgent(input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=0.1, target_update=10)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state_tensor)
        next_state, reward, done, _ = env.step(action.item())
        print(f"Done: {done}") 
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0) if not done else None
        agent.memory.push(state, action, next_state, torch.tensor([reward], dtype=torch.float32), torch.tensor([done], dtype=torch.bool))
        state = next_state
        agent.optimize_model()
        env.render()
    if episode % agent.target_update == 0:
        agent.update_target_network()
    print('Episode {}: Total Reward = {}'.format(episode, total_reward))

# Save trained model
torch.save(agent.policy_net.state_dict(), 'dqn_model.pth')

# Close environment
env.close()

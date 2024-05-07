import matplotlib.pyplot as plt

# Read data from file
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

import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='Scenario_train.txt', level=logging.INFO, format='%(message)s')

survived_count = 0
collided_count = 0

# Read data from file
file_path = 'training_log.txt'
with open(file_path, 'r') as file:
    data = file.readlines()

k = 0
# Iterate through each line and count occurrences
for line in data:
    if (k%100 == 0 and k >0):
        logging.info('Scenario {}: survived_count = {}, collided_count: {}'.format((k/100), survived_count, collided_count))
    if "Survived" in line:
        survived_count += 1
    elif "Collided" in line:
        collided_count += 1
    k += 1

logging.info('Scenario {}: survived_count = {}, collided_count: {}'.format((12), survived_count, collided_count))
# Print the counts
print("Survived count:", survived_count)
print("Collided count:", collided_count)
print("Accuracy:", survived_count/(survived_count + collided_count))
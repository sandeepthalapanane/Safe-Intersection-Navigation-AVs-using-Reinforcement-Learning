import os, sys, subprocess
import traci

# Define the port you want to use (choices: 8813, 8843, 8873, etc.)
# PORT = 8870

def initSimulator(withGUI, portnum, name):

	# Path to the sumo binary
	if withGUI:
		sumoBinary = "C:/Sumo/bin/sumo-gui"
	else:
		sumoBinary = "C:/Sumo/bin/sumo"

	# Load the scenario
	# descriptor = descriptor.astype(int)
	task = str(name)+'.sumo'+ '.cfg'
	sumoConfig = "C:/Users/sande/Desktop/Safe-Intersection-Navigation-AVs-using-Reinforcement-Learning/gym_sumo/envs/sumo_networks/"+task

	# Call the sumo simulator
	sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
		"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), \
		"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

	# Initialize the simulation
	traci.init(portnum)
	return traci

def closeSimulator(traci):
	traci.close()
	sys.stdout.flush()


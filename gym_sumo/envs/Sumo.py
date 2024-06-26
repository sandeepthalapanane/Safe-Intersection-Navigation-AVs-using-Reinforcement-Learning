import os, sys, subprocess
import traci
import argparse

# Define the port you want to use (choices: 8813, 8843, 8873, etc.)
# PORT = 8870

def initSimulator(withGUI, portnum, name):

	# Path to the sumo binary
	if withGUI:
		sumoBinary = "C:/Sumo/bin/sumo-gui"
		# sumoBinary = "D:/CMSC473/SUMO/bin/sumo-gui.exe"
	else:
		sumoBinary = "C:/Sumo/bin/sumo"
		# sumoBinary = "D:/CMSC473/SUMO/bin/sumo"


	# Load the scenario
	# descriptor = descriptor.astype(int)
	task = str(name)+'.sumo'+ '.cfg'
	cwd = os.getcwd()
	sumoConfigPath = os.path.join(cwd, "gym_sumo/envs/sumo_networks/")
	# sumoConfig = "C://Users//sande//Desktop/Safe-Intersection-Navigation-AVs-using-Reinforcement-Learning//gym_sumo//envs//sumo_networks//"+task
	sumoConfig = sumoConfigPath+task

	# Call the sumo simulator
	sumoProcess = subprocess.Popen([sumoBinary, "-c", sumoConfig, "--remote-port", str(portnum), \
		"--time-to-teleport", str(-1), "--collision.check-junctions", str(True), "--eager-insert", str(True), \
		"--no-step-log", str(True), "--no-warnings", str(True)], stdout=sys.stdout, stderr=sys.stderr)

	# Initialize the simulation
	traci.init(portnum)
	return traci

def closeSimulator(traci):
	traci.close()
	sys.stdout.flush()


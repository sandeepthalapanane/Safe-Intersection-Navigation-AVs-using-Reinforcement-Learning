import os, subprocess, time, signal

import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import sys
import numpy as np
import matplotlib.pyplot as plt

from gym_sumo.envs import Sumo
import random

import logging
logger = logging.getLogger(__name__)


np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.random.seed(1)
scenario_counter = 0

class Car:
	""" A class struct that stores the car features.
	"""
	def __init__(self, carID, position = None, distance = None, speed = 0, angle = None, signal = None, length = None):
		self.carID = carID
		self.position = position
		self.distance = distance
		self.speed = speed
		self.angle = angle
		self.signal = signal
		self.length = length

class SumoEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		## SIMULATOR SETTINGS
		self.withGUI = True #True

		if self.withGUI:
			print("Press Ctrl-A to start simulation")
		# 0. Left, Center, Right
		# 1. # Lanes
		# 2. Density of Traffic
		# 3. Angles
		# self.scenario = np.asarray([0,2,2,1])
		# self.scenarios_list = scenarios_list
		# print(scenarios_list)
		self.scenarios_list = [13, 14]
		# random.shuffle(self.scenarios_list)
		self.scenario = self.scenarios_list[scenario_counter]
		# print(self.scenario)
		self.observation = []
		self.port_number = 8870
		self.traci = Sumo.initSimulator(self.withGUI, self.port_number, self.scenario)
		self.dt = self.traci.simulation.getDeltaT()/1000.0
		
		## INITIALIZE EGO CAR
		self.egoCarID = 'veh0'
		self.max_speed = 20.1168		# m/s 
		self.observation = self.reset()

		## SET OPENAI VALS
		self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)), dtype=np.float64)
		self.action_space = spaces.Discrete(5)
	
	
	def scenario_counter(self, epsiode):
		
		global scenario_counter
		scenario_counter = int(epsiode/20)
		self.scenario = self.scenarios_list[scenario_counter]
		if self.traci is not None:
			self.traci.close()
			sys.stdout.flush()
			self.traci = None
			self.traci = Sumo.initSimulator(self.withGUI, self.port_number, self.scenario)
    	
		print("Scenario selected: ", self.scenario)
		print("Scenario Counter: ", scenario_counter)

		return scenario_counter
	

	def step(self, action):
		## Dynamic Frame Skipping
		if action==0:
			act = 0 #'go'
			action_steps = 1
		else:
			act = 1 #'wait'
			action_steps = 2**(action-1)
		
		# Take step
		r = 0
		for a in range(action_steps): 
			self.takeAction(act)
			self.traci.simulationStep()

			# Get reward and check for terminal state
			reward, terminal, terminalType = self.get_reward()
			r += reward

			braking = self.isTrafficBraking()
			# if egoCar.isTrafficWaiting(): waitingTime += 1

		self.observation = self.getFeatures()
		info = {braking, terminalType}

		return (self.observation, reward, terminal, {})

	def get_reward(self):
		""" Reward function for the domain and check for terminal state."""
		terminal = False
		terminalType = 'None'

		try:
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			distance_ego = np.asarray([np.linalg.norm(position_ego - self.endPos)])
			distance_ego = distance_ego[0]
		except:
			print("traci couldn't find car") ################################################# should this be a collision?
			return -1.0, True, 'Car not found'
			distance_ego = 0

		# Step cost
		reward = -.01 

		# Collision check
		teleportIDList = self.traci.simulation.getStartingTeleportIDList()
		if teleportIDList:
			collision = True
			reward = -1.0 
			terminal = True
			terminalType = 'Collided!!!'
			 
			

		else: # Goal check
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID))
			distance_ego = np.linalg.norm(position_ego - self.endPos)
			thresh = 5
			if self.scenario in self.scenarios_list:
				if distance_ego <= thresh:
					reward = 1.0 
					terminal = True
					terminalType = 'Survived'
					print(terminalType)
			

		if (terminalType == 'Collided!!!'):
			print(terminalType)
			print(terminal)
		
		return reward, terminal, terminalType


	def reset(self, seed=None):
		""" Repeats NO-OP action until a new episode begins. """
		try:
			self.traci.vehicle.remove(self.egoCarID)
		except:
			pass

		super().reset(seed=seed)
		self.addEgoCar()            # Add the ego car to the scene
		self.setGoalPosition()      # Set the goal position
		self.traci.simulationStep() # Take a simulation step to initialize car

		self.observation = self.getFeatures()
		info = self._getInfo()
		
		return self.observation, info #

	def render(self, mode='human', close=False):
		""" Viewer only supports human mode currently. """

		im = np.flipud(self.observation)
		if np.size(im)>0:
			plt.clf()
			# rang = np.max(im.ravel())-np.min(im.ravel())
			# im = (im - np.min(im.ravel()))/rang
			im = np.minimum(im,1)
			im = np.maximum(im,0)
			plt.imshow(im, interpolation='nearest')
			plt.show(block=False)
			plt.pause(0.000001)
		return im
		# print(self.scenario)


	def setGoalPosition(self):
		""" Set the agent goal position depending on the training scenario.
		Note this is only getReward only checks X, 
		""" 

		# 0. Left, Center, Right
		# 1. # Lanes
		if self.scenario==1: # left
			self.endPos = [122.94, 119.26]
		elif self.scenario==2:
			self.endPos = [176.07, 61.77]
		elif self.scenario==3:
			self.endPos = [77.29, 43.40]
		elif self.scenario==4:
			self.endPos = [94.45, 80.45]
		elif self.scenario==5:   
			self.endPos = [171.76, 201.74]
		elif self.scenario==6:
			self.endPos = [103.28, 126.17]
			#self.endPos = [115.0, 95.0]
		elif self.scenario==7:
			self.endPos = [101.61, 106.89]
		elif self.scenario==8:
			self.endPos = [153.35, 87.09]
		elif self.scenario==9:   
			self.endPos = [121.96, 152.00]
		elif self.scenario==10:   
			self.endPos = [94.32, 94.12]
		elif self.scenario==11:   
			self.endPos = [224.60, 153.65]
		elif self.scenario==12:   
			self.endPos = [209.49, 82.01]
		elif self.scenario==13:
			self.endPos = [530.59, 292.77]
		elif self.scenario==14:
			self.endPos = [352.73, 365.63]
			

	def _getInfo(self):
		return {"current_episode":0}

	def addEgoCar(self):																

		vehicles=self.traci.vehicle.getIDList()

		## PRUNE IF TRAFFIC HAS BUILT UP TOO MUCH
		# if more cars than setnum, p(keep) = setnum/total
		setnum = 20
		if len(vehicles)>0:
			keep_frac = float(setnum)/len(vehicles)
		for i in range(len(vehicles)):
			if vehicles[i] != self.egoCarID:
				if np.random.uniform(0,1,1)>keep_frac:
					self.traci.vehicle.remove(vehicles[i])

		## DELAY ALLOWS CARS TO DISTRIBUTE 
		for j in range(np.random.randint(40,50)):#np.random.randint(0,10)):
			self.traci.simulationStep()

		## STARTING LOCATION
		# depart = -1   (immediate departure time)
		# pos    = -2   (random position)
		# speed  = -2   (random speed)
		if self.scenario==1: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=76.47, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==2: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=47.50, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==3: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=39.36, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==4: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=81.49, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==5: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=108.26, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==6: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=84.65, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==7: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=83.58, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==8: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=76.47, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==9: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=108.26, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==10: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=76.47, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==11: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=43.05, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==12: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=48.80, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==13: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=70.27, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==14: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=39.2, departSpeed=0, departLane=0, typeID='vType0')
		self.traci.vehicle.setSpeedMode(self.egoCarID, int('00000', 2))
		



	def takeAction(self, action):
		""" Take the action following the vehicle dynamics and check for bounds.
		"""

		# action 0 = go
		# action 1 = stay
		if action == 0: #accelerate
			accel = 450
			speed = 15
		elif action == 1: # wait
			accel = 0
			speed = 0
		
		# New speed
		speed = speed + self.dt*accel
		# Lower and upper bound for speed on straight roads and the turn
		if speed < 0.0:
			# Below zero
			speed = 0.0
		elif speed > self.max_speed:
			# Exceeded lane speed limit
			speed = self.max_speed
		speed = speed
		self.traci.vehicle.slowDown(self.egoCarID, speed, 0) #int(self.dt*1000)) 
		#self.traci.vehicle.setAccel(self.egoCarID, accel) # should allow negative speeds
		

	def getFeatures(self):
		""" Main file for ego car features at an intersection.
		"""

		carDistanceStart, carDistanceStop, carDistanceNumBins = 0, 80, 40
		## LOCAL (101, 90ish)
		carDistanceYStart, carDistanceYStop, carDistanceYNumBins = -5, 40, 18 # -4, 24, relative to ego car
		carDistanceXStart, carDistanceXStop, carDistanceXNumBins = -80, 80, 26
		TTCStart, TTCStop, TTCNumBins = 0, 6, 30    # ttc
		carSpeedStart, carSpeedStop, carSpeedNumBins = 0, 20, 10 # 20  
		carAngleStart, carAngleStop, carAngleNumBins = -180, 180, 10 #36

		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)
		ego_angle = self.traci.vehicle.getAngle(self.egoCarID)
		ego_v = self.traci.vehicle.getSpeed(self.egoCarID)

	
		discrete_features = np.zeros((carDistanceYNumBins, carDistanceXNumBins,3))

		# ego car
		pos_x_binary = self.getBinnedFeature(0, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
		pos_y_binary = self.getBinnedFeature(0, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
		x = np.argmax(pos_x_binary)
		y = np.argmax(pos_y_binary)
		discrete_features[y,x,:] = [0.0, ego_v/20.0, 1]

		for carID in self.traci.vehicle.getIDList(): 
			if carID==self.egoCarID:
				continue
			c_x,c_y = self.traci.vehicle.getPosition(carID)
			angle = self.traci.vehicle.getAngle(carID) 
			c_v = self.traci.vehicle.getSpeed(carID)
			c_vx = c_v*np.sin(np.deg2rad(angle))
			c_vy = c_v*np.cos(np.deg2rad(angle))
			#print 'angle', angle, np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))
			zx,zv = c_x,c_vx
			
			#position
			p_x = c_x-ego_x
			p_y = c_y-ego_y
			# angle
			carframe_angle = wrapPi(angle-ego_angle)
			
			#print angle, carframe_angle
			c_vec = np.asarray([p_x, p_y, 1])
			rot_mat = np.asarray([[ np.cos(np.deg2rad(ego_angle)), np.sin(np.deg2rad(ego_angle)), 0],
								  [-np.sin(np.deg2rad(ego_angle)), np.cos(np.deg2rad(ego_angle)), 0],
								  [                             0,                             0, 1]])
			rot_c = np.dot(c_vec,rot_mat) 

			carframe_x = rot_c[0] 
			carframe_y = rot_c[1] 
			
			# f = [carframe_x, carframe_y, carframe_angle, c_v] 
			# features.append(f)

			pos_x_binary = self.getBinnedFeature(carframe_x, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
			pos_y_binary = self.getBinnedFeature(carframe_y, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
			
			x = np.argmax(pos_x_binary)
			y = np.argmax(pos_y_binary)

			discrete_features[y,x,:] = [carframe_angle/90.0, c_v/20.0, 1]
			
		
		return discrete_features
 
	# def getFeatures(self):
	# 		""" Main function for ego car features at an intersection. """
	# 		carDistanceStart, carDistanceStop, carDistanceNumBins = 0, 80, 40
	# 		carDistanceYStart, carDistanceYStop, carDistanceYNumBins = -5, 40, 18
	# 		carDistanceXStart, carDistanceXStop, carDistanceXNumBins = -80, 80, 26
	# 		TTCStart, TTCStop, TTCNumBins = 0, 6, 30
	# 		carSpeedStart, carSpeedStop, carSpeedNumBins = 0, 20, 10
	# 		carAngleStart, carAngleStop, carAngleNumBins = -180, 180, 10

	# 		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)
	# 		ego_angle = self.traci.vehicle.getAngle(self.egoCarID)
	# 		ego_v = self.traci.vehicle.getSpeed(self.egoCarID)

	# 		# Initialize the state vector
	# 		state_vector = []

	# 		# Add ego car features
	# 		state_vector.append(ego_v/20.0)  # Normalized ego car speed

	# 		# Add features for other vehicles
	# 		for carID in self.traci.vehicle.getIDList():
	# 			if carID == self.egoCarID:
	# 				continue
	# 			c_x, c_y = self.traci.vehicle.getPosition(carID)
	# 			angle = self.traci.vehicle.getAngle(carID)
	# 			c_v = self.traci.vehicle.getSpeed(carID)
	# 			c_vx = c_v * np.sin(np.deg2rad(angle))
	# 			c_vy = c_v * np.cos(np.deg2rad(angle))

	# 			# Calculate relative position and angle
	# 			p_x = c_x - ego_x
	# 			p_y = c_y - ego_y
	# 			carframe_angle = wrapPi(angle - ego_angle)

	# 			# Add features to the state vector
	# 			state_vector.append(carframe_angle / 90.0)  # Normalized relative angle
	# 			state_vector.append(c_v / 20.0)             # Normalized relative speed

	# 		# Convert the state vector to a numpy array
	# 		state_array = np.array(state_vector, dtype=np.float32)
	# 		print(state_array.shape)
	# 		return state_array


	def getBinnedFeature(self, val, start, stop, numBins):
		""" Creating binary features.
		"""
		bins = np.linspace(start, stop, numBins)
		binaryFeatures = np.zeros(numBins)

		if val == 'unknown':
			return binaryFeatures

		# Check extremes
		if val <= bins[0]:
			binaryFeatures[0] = 1
		elif val > bins[-1]:
			binaryFeatures[-1] = 1

		# Check intermediate values
		for i in range(len(bins) - 1):
			if val > bins[i] and val <= bins[i+1]:
				binaryFeatures[i+1] = 1

		return binaryFeatures

	def isTrafficBraking(self):
		""" Check if any car is braking
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				brakingState = self.traci.vehicle.getSignals(carID)
				if brakingState == 8:
					return True
		return False

	def isTrafficWaiting(self):
		""" Check if any car is waiting
		"""
		for carID in self.traci.vehicle.getIDList():
			if carID != self.egoCarID:
				speed = self.traci.vehicle.getSpeed(carID)
				if speed <= 1e-1:
					return True
		return False

def wrapPi(angle):
	# makes a number -pi to pi
	while angle <= -180:
		angle += 360
	while angle > 180:
		angle -= 360
	return angle

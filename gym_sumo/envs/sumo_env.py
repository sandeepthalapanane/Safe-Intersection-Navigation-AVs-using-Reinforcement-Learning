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

	def __init__(self, mode='train', **kwargs):
		self.mode = mode
		if self.mode == 'train':
			self.scenarios_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
			self.withGUI = False
		elif self.mode == 'test':
			self.scenarios_list = [13, 14]
			self.withGUI = True
		

		if self.withGUI:
			print("Press Ctrl-A to start simulation")

		self.scenario = self.scenarios_list[scenario_counter]
		self.observation = []
		self.q = 0
		self.port_number = 8870
		self.traci = Sumo.initSimulator(self.withGUI, self.port_number, self.scenario)
		self.dt = self.traci.simulation.getDeltaT()/1000.0
		
		self.egoCarID = 'veh0'
		self.max_speed = 20.1168		
		self.observation = self.reset()

		self.observation_space = spaces.Box(low=0, high=1, shape=(np.shape(self.observation)), dtype=np.float64)
		self.action_space = spaces.Discrete(5)
	
	
	def scenario_counter(self, index):
		
		self.scenario = self.scenarios_list[index]
		if self.traci is not None:
			self.traci.close()
			sys.stdout.flush()
			self.traci = None
			self.traci = Sumo.initSimulator(self.withGUI, self.port_number, self.scenario)
    	
		print("Scenario selected: ", self.scenario)
		print("Scenario Counter: ", index)

		return 
	

	def step(self, action):

		if action==0:
			act = 0 #'go'
			action_steps = 1
		else:
			act = 1 #'wait'
			action_steps = 2**(action-1)

		r = 0
		for a in range(action_steps): 
			self.takeAction(act)
			self.traci.simulationStep()

			
			reward, terminal, terminalType = self.get_reward()
			r += reward

			braking = self.isTrafficBraking()
			# if self.isTrafficWaiting(): waitingTime += 1

		self.observation = self.getFeatures()
		info = {'information': terminalType}

		return (self.observation, r, terminal, info)

	def get_reward(self):
		""" Reward function for the domain and check for terminal state."""
		terminal = False
		terminalType = 'None'

		try:
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID), dtype=np.float32)
			distance_ego = np.asarray([np.linalg.norm(position_ego - self.endPos)], dtype=np.float32)
			distance_ego = distance_ego[0]
		except:
			print("traci couldn't find car") 
			return -1.0, True, 'Car not found'
			distance_ego = 0

		reward = -.01 


		teleportIDList = self.traci.simulation.getStartingTeleportIDList()
		if teleportIDList:
			collision = True
			reward = -1.0 
			terminal = True
			terminalType = 'Collided!!!'
			 
			

		else:
			position_ego = np.asarray(self.traci.vehicle.getPosition(self.egoCarID), dtype=np.float32)
			distance_ego = np.linalg.norm(position_ego - self.endPos)
			thresh = 10
			if self.scenario in self.scenarios_list:
				if distance_ego <= thresh:
					reward = 1.0 
					terminal = True
					terminalType = 'Survived'

		return reward, terminal, terminalType


	def reset(self, seed=None):	
		""" Repeats NO-OP action until a new episode begins. """
		try:
			self.traci.vehicle.remove(self.egoCarID)
		except:
			pass

		super().reset(seed=seed)
		
		self.addEgoCar()            
		self.setGoalPosition()      
		self.traci.simulationStep() 
		self.observation = self.getFeatures()
		info = self._getInfo()
		
		return self.observation, info 

	def render(self, mode='human', close=False):
		""" Viewer only supports human mode currently. """

		im = np.flipud(self.observation)
		if np.size(im)>0:
			plt.clf()
			im = np.minimum(im,1)
			im = np.maximum(im,0)
			plt.imshow(im, interpolation='nearest')
			plt.show(block=False)
			plt.pause(0.000001)
		return im



	def setGoalPosition(self):
		""" Set the agent goal position depending on the training scenario.
		Note this is only getReward only checks X, 
		""" 

		if self.scenario==1: 
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
			self.endPos = [188.56, 82.01]
		elif self.scenario==13:
			self.endPos = [530.59, 292.77]
		elif self.scenario==14:
			self.endPos = [352.73, 365.63]
			

	def _getInfo(self):
		return {"current_episode":0}

	def addEgoCar(self):																

		vehicles=self.traci.vehicle.getIDList()


		setnum = 20
		if len(vehicles)>0:
			keep_frac = float(setnum)/len(vehicles)
		for i in range(len(vehicles)):
			if vehicles[i] != self.egoCarID:
				if np.random.uniform(0,1,1)>keep_frac:
					self.traci.vehicle.remove(vehicles[i])

		for j in range(np.random.randint(40,50)):
			self.traci.simulationStep()

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
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=107.54, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==12: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=56.25, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==13: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=70.27, departSpeed=0, departLane=0, typeID='vType0')
		if self.scenario==14: 
			self.traci.vehicle.add(self.egoCarID, 'routeEgo', depart="0", departPos=39.2, departSpeed=0, departLane=0, typeID='vType0')
		self.traci.vehicle.setSpeedMode(self.egoCarID, int('00000', 2))
		



	def takeAction(self, action):
		""" Take the action following the vehicle dynamics and check for bounds.
		"""

		if action == 0: #accelerate
			accel = 17.88
			speed = 15
		elif action == 1: # wait
			accel = 0
			speed = 0
		
		
		speed = speed + self.dt*accel
		
		if speed < 0.0:
			speed = 0.0
		elif speed > self.max_speed:
			speed = self.max_speed
		

		self.traci.vehicle.slowDown(self.egoCarID, speed, 2) 
		

	def getFeatures(self):
		""" Main file for ego car features at an intersection.
		"""


		carDistanceYStart, carDistanceYStop, carDistanceYNumBins = -5, 40, 18 
		carDistanceXStart, carDistanceXStop, carDistanceXNumBins = -80, 80, 26

		ego_x, ego_y = self.traci.vehicle.getPosition(self.egoCarID)
		ego_angle = self.traci.vehicle.getAngle(self.egoCarID)
		ego_v = self.traci.vehicle.getSpeed(self.egoCarID)

	
		discrete_features = np.zeros((carDistanceYNumBins, carDistanceXNumBins,3)).astype(np.float32)

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
			zx,zv = c_x,c_vx
			
			#position
			p_x = c_x-ego_x
			p_y = c_y-ego_y
			# angle
			carframe_angle = wrapPi(angle-ego_angle)
			
			#print angle, carframe_angle
			c_vec = np.asarray([p_x, p_y, 1], dtype=np.float32)
			rot_mat = np.asarray([[ np.cos(np.deg2rad(ego_angle)), np.sin(np.deg2rad(ego_angle)), 0],
								  [-np.sin(np.deg2rad(ego_angle)), np.cos(np.deg2rad(ego_angle)), 0],
								  [                             0,                             0, 1]], dtype=np.float32)
			rot_c = np.dot(c_vec,rot_mat) 

			carframe_x = rot_c[0] 
			carframe_y = rot_c[1] 

			pos_x_binary = self.getBinnedFeature(carframe_x, carDistanceXStart, carDistanceXStop, carDistanceXNumBins)
			pos_y_binary = self.getBinnedFeature(carframe_y, carDistanceYStart, carDistanceYStop, carDistanceYNumBins)
			
			x = np.argmax(pos_x_binary)
			y = np.argmax(pos_y_binary)

			discrete_features[y,x,:] = [carframe_angle/90.0, c_v/20.0, 1]
			
		
		return discrete_features

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

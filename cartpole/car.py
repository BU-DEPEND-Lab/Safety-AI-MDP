import numpy as np
import random
from math import *

class car:
	def __init__(self, states, state = None):
		self.states=np.array(states)
		self.trajectory=[]
		self.transitions = None
		if state is None:
			self.state = self.states[len(self.states)/2, len(self.states[len(self.states)/2])/2]

	def move(self, transitions, action):
		self.trajectory.append([self.state, action])
		probability = random.random()
		for i in range(len(transitions[self.state[0], self.state[1], action])):
			for j in range(len(transitions[self.state[0], self.state[1], action, i])):
				probability = probability - transitions[self.state[0], self.state[1], action, i, j]
				if probability <= 0:
					self.state = self.states[i, j]
					return self.state 
		return self.state 


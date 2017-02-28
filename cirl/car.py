import numpy as np
import random
from math import *

class car:
	def __init__(self, states=None, probability=0.3):
		self.states=states
		self.state=self.states[0,0].astype(int)
		self.trajectory=[]
		self.probability=probability

	def move(self, action=0):
		self.trajectory.append([self.state, action])
		if self.state[0]==0 or self.state[0] == len(self.states) -1 or self.state[1]==0 or self.state[1]==len(self.states)-1:
			return self.state
		p=random.random()
		if p < self.probability:
			action=random.randint(0, 3)
			#new_action=random.randint(0, 3) 
			#if new_action>=action:
			#	new_action=new_action+1
			#action=new_action
		if action==0:
			self.state=self.state
		elif action==2:
			if self.state[1]<len(self.states[self.state[0]])-1:
				self.state=self.states[self.state[0], self.state[1]+1].astype(int);
			else:
				self.state=self.state
		elif action==1:
			if self.state[0]<len(self.states)-1:
				self.state=self.states[self.state[0]+1, self.state[1]].astype(int);
			else:
				self.state=self.state
		elif action==4:
			if self.state[1]>0:
				self.state=self.states[self.state[0], self.state[1]-1].astype(int);
			else:
				self.state=self.state
		elif action==3:
			if self.state[0]>0:
				self.state=self.states[self.state[0]-1, self.state[1]].astype(int);
			else:
				self.state=self.state
		return self.state	


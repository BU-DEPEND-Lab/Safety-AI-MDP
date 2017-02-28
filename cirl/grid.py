import numpy as np
from math import *

class grid():
	def __init__(self, x_max=10, y_max=10):
		self.x_max=x_max
		self.y_max=y_max
		self.r_max=int(self.x_max**2+self.y_max**2)
		self.r_avg=int(self.r_max/2)
		self.r_min=0
		self.loc_max_0=[int(self.y_max/3), int(self.x_max/3)]
		self.loc_max_1=[int(self.y_max/3), int(self.x_max*2/3)]
		self.loc_min_0=[int(self.y_max*2/3), int(self.x_max/2)]
		self.features=np.zeros([self.y_max, self.x_max, 3])
		self.rewards=np.zeros([self.y_max, self.x_max])
		self.states = np.zeros([self.y_max, self.x_max, 2])
		self.transitions = np.zeros([self.y_max, self.x_max, 5, self.y_max, self.x_max])
		self.probability = 0.7
		for i in range(len(self.states)):
			for j in range(len(self.states[i])):
				self.states[i, j, 0]=i
				self.states[i, j, 1]=j
	 
	##	self.theta=np.array(theta) / np.linalg.norm(np.array(theta))

		self.feature_1=lambda position:  (self.r_avg+(self.r_max-self.r_avg)*exp(-sqrt((position[0]-self.loc_max_0[0])**2+(position[1]-self.loc_max_0[1])**2)))/self.r_max
		self.feature_2=lambda position:  (self.r_avg+(self.r_max-self.r_avg)*exp(-sqrt((position[0]-self.loc_max_1[0])**2+(position[1]-self.loc_max_1[1])**2)))/self.r_max
		self.feature_3=lambda position:  (self.r_avg+(self.r_min-self.r_avg)*exp(-sqrt((position[0]-self.loc_min_0[0])**2+(position[1]-self.loc_min_0[1])**2)))/self.r_max

		for i in range(0, self.y_max):
			for j in range(0, self.x_max):
				self.features[i, j]= [self.feature_1([i, j]), self.feature_2([i, j]), self.feature_3([i, j])]

		for i in range(len(self.transitions)):
			for j in range(len(self.transitions[i])):
				for k in range(5):
					self.transitions[i, j, k] = np.zeros([self.y_max, self.x_max])
					if i == 0 or i == self.y_max-1 or j == 0 or j == self.x_max - 1:
						self.transitions[i, j, k, i, j] = 1
						next
					else:
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/5
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/5
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/5
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/5
						self.transitions[i, j, k, i, j]   = (1 - self.probability)/5
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j+1] = self.transitions[i, j, k, i, j+1]  + self.probability
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i+1, j]  + self.probability
							




	#initialize a grid, all rewards are settled in each coord
	def w_features(self, theta):
		theta=theta/np.linalg.norm(theta)
		for i in range(len(self.features)):
			for j in range(len(self.features[i])):
				self.rewards[i, j]=np.dot(np.array(theta), np.transpose(self.features[i, j]))
	
	

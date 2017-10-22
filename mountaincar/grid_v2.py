import numpy as np
from math import *

class grid():
	def __init__(self, x_max=10, y_max=10, actions = 5, feature_states = np.random.randint(10, size = [2, 2, 2]), probability = 0.7, bounce=True, sink= True):
		self.x_max=int(x_max)
		self.y_max=int(y_max)
		self.actions = int(actions)
		self.r_max=int(self.x_max**2+self.y_max**2)
		self.r_avg=int(self.r_max/2)
		self.r_min=int(0)
		self.feature_states = feature_states
##################
		
		
##################
		self.features=np.zeros([self.y_max, self.x_max, np.array(feature_states).size/2]).astype(float)
		self.rewards=np.zeros([self.y_max, self.x_max]).astype(float)
		self.states = np.zeros([self.y_max, self.x_max, 2]).astype(int)
		self.transitions = np.zeros([self.y_max, self.x_max, self.actions, self.y_max, self.x_max]).astype(float)
		self.probability = float(probability)
		for i in range(len(self.states)):
			for j in range(len(self.states[i])):
				self.states[i, j, 0]=i
				self.states[i, j, 1]=j

		for i in range(self.y_max):
			for j in range(self.x_max):
				index = 0
				for l in range(0, len(feature_states)):
					for m in range(0, len(feature_states[l])):
						self.features[i, j, index + m] = exp(-1.0 * sqrt((i - feature_states[l][m][0])**2 + (j - feature_states[l][m][1])**2))
					index  = index + len(feature_states[l])

		for i in range(len(self.transitions)):
			for j in range(len(self.transitions[i])):
				for k in range(len(self.transitions[i, j])):
					if k == 0:
						self.transitions[i, j, k, i, j] += 1.0
					else:
						self.transitions[i, j, k, self.y_max - 1 - abs(self.y_max -1 - (i+1)), j] +=  (1.0 - self.probability)/4.0
						self.transitions[i, j, k, i, self.x_max - 1 - abs(self.x_max -1 - (j+1))] += (1.0 - self.probability)/4.0
						self.transitions[i, j, k, abs(i-1), j] +=(1.0 - self.probability)/4.0
						self.transitions[i, j, k, i, abs(j-1)] +=(1.0 - self.probability)/4.0
					###
						if k == 1:
							self.transitions[i, j, k, i, self.x_max -1 - abs(self.x_max -1 - (j+1))] +=  self.probability	
						elif k == 2:
							self.transitions[i, j, k, self.y_max - 1 - abs(self.y_max -1 - (i+1)), j] += self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, abs(j-1)] += self.probability
						elif k == 4:
							self.transitions[i, j, k, abs(i-1), j] += self.probability		
						next
					

		if sink is True:
			for k in range(self.actions):
				#self.transitions[self.loc_min_0[0], self.loc_min_0[1], k] = np.zeros([y_max, x_max])
				#self.transitions[self.loc_min_0[0], self.loc_min_0[1], k, self.loc_min_0[0], self.loc_min_0[1]] = 1.0
				for high in feature_states[0]:
					self.transitions[high[0], high[1], k] = np.zeros([y_max, x_max])
					self.transitions[high[0], high[1], k, high[0], high[1]] = 1.0
		

		file = open('state_space', 'w')
		file.write("x_max\n")
		file.write(str(self.x_max))
		file.write("\ny_max\n")
		file.write(str(self.y_max))
		file.write("\nactions\n")
		file.write(str(self.actions))
		for high in range(len(feature_states[0])):
			file.write("\nx_h_" + str(high) + "\n")
			file.write(str(feature_states[0][high][0]))
			file.write("\ny_h_" + str(high) + "\n")
			file.write(str(feature_states[0][high][1]))
		for low in range(len(feature_states[1])):
			file.write("\nx_l_" + str(low) + "\n")
			file.write(str(feature_states[1][low][0]))
			file.write("\ny_l_" + str(low) + "\n")
			file.write(str(feature_states[1][low][1]))
		file.close()
		

	def w_features(self, theta):
		theta=theta/np.linalg.norm(theta)
		for i in range(self.y_max):
			for j in range(self.x_max):
				self.rewards[i, j]=np.dot(np.array(theta), np.transpose(self.features[i, j]))
	
		return self.rewards
	
	def pi_transitions(self, policy, probability = None):
		if probability is None:
			probability = self.probability
		P = np.zeros([self.y_max, self.x_max, self.y_max, self.x_max]).astype(float)
		for y in range(self.y_max):
			for x in range(self.x_max):
				for y_ in range(self.y_max):
					for x_ in range(self.x_max):
						for action in range(self.actions):
							if action == int(policy[y, x]):
								P[y, x, y_, x_] += self.transitions[y, x, action, y_, x_] * (probability + (1 - probability) / self.actions)
							else:
								P[y, x, y_, x_] += self.transitions[y, x, action, y_, x_] * (1 - probability) / self.actions
		return P		 

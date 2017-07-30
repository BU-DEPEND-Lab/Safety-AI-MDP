import numpy as np
from math import *

class grid():
	def __init__(self, x_max=10, y_max=10, random = True, feature_states = None, probability = 0.7, bounce=True, sink= True):
		self.x_max=int(x_max)
		self.y_max=int(y_max)
		self.r_max=int(self.x_max**2+self.y_max**2)
		self.r_avg=int(self.r_max/2)
		self.r_min=int(0)
##################
		if random is False:
			if feature_states is None:
				self.loc_min_0=[int(self.y_max/3), int(self.x_max/3)]
				self.loc_max_1=[int(self.y_max/3), int(self.x_max*2/3)]
				self.loc_max_0=[int(self.y_max*2/3), int(self.x_max/2)]
				self.loc_min_1=[int(self.y_max/2), int(self.y_max/2)]
			else:
				self.loc_max_0=np.array(feature_states[0])
				self.loc_max_1=np.array(feature_states[1])
				self.loc_min_0=np.array(feature_states[2])
				self.loc_min_1=np.array(feature_states[3])

		else:
			self.loc_min_0 = np.random.randint(y_max, size=2)
			self.loc_max_1 = np.random.randint(y_max, size=2)
			self.loc_max_0 = np.random.randint(y_max, size=2)
			self.loc_min_1 = np.random.randint(y_max, size=2)
	
##################
		self.features=np.zeros([self.y_max, self.x_max, 4]).astype(float)
		self.rewards=np.zeros([self.y_max, self.x_max]).astype(float)
		self.states = np.zeros([self.y_max, self.x_max, 2]).astype(int)
		self.transitions = np.zeros([self.y_max, self.x_max, 5, self.y_max, self.x_max]).astype(float)
		self.probability = float(probability)
		for i in range(len(self.states)):
			for j in range(len(self.states[i])):
				self.states[i, j, 0]=i
				self.states[i, j, 1]=j
	 
		self.feature_1=lambda position:  exp(-1.0 * sqrt((position[0]-self.loc_max_0[0])**2+(position[1]-self.loc_max_0[1])**2))
		self.feature_2=lambda position:  exp(-1.0 * sqrt((position[0]-self.loc_max_1[0])**2+(position[1]-self.loc_max_1[1])**2))
		self.feature_3=lambda position:  exp(-1.0 * sqrt((position[0]-self.loc_min_0[0])**2+(position[1]-self.loc_min_0[1])**2))
		self.feature_4=lambda position:  exp(-1.0 * sqrt((position[0]-self.loc_min_1[0])**2+(position[1]-self.loc_min_1[1])**2))

		for i in range(0, self.y_max):
			for j in range(0, self.x_max):
				self.features[i, j]= [self.feature_1([i, j]), self.feature_2([i, j]), self.feature_3([i, j]), self.feature_4([i, j])]

		for i in range(len(self.transitions)):
			for j in range(len(self.transitions[i])):
				for k in range(5):
					self.transitions[i, j, k] = np.zeros([self.y_max, self.x_max])
					###
					if k == 0:
						self.transitions[i, j, k, i, j] = 1.0
					else:
						self.transitions[i, j, k, y_max -1 - abs(y_max -1 - (i+1)), j] =  (1.0 - self.probability)/4.0
						self.transitions[i, j, k, i, x_max - 1 - abs(x_max -1 - (j+1))] = (1.0 - self.probability)/4.0
						self.transitions[i, j, k, abs(i-1), j] =(1.0 - self.probability)/4.0
						self.transitions[i, j, k, i, abs(j-1)] =(1.0 - self.probability)/4.0
					###
						if k == 1:
							self.transitions[i, j, k, i, x_max -1 - abs(x_max -1 - (j+1))] = self.transitions[i, j, k, i,  x_max -1 - abs(x_max -1 - (j+1))] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, y_max - 1 - abs(y_max -1 - (i+1)), j] = self.transitions[i, j, k, y_max -1 - abs(y_max -1 - (i+1)), j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, abs(j-1)] = self.transitions[i, j, k, i, abs(j-1)]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, abs(i-1), j] = self.transitions[i, j, k, abs(i-1), j]  + self.probability		
						next
					

					'''
					if i == 0 and j == 0:
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/3
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/3
						self.transitions[i, j, k, i, j] = (1 - self.probability)/3
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j+1, k, i, j] = self.transitions[i, j+1, k, i, j] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability

					elif i == 0 and j == self.x_max-1:
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/3
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/3
						self.transitions[i, j, k, i, j] = (1 - self.probability)/3
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability


					elif i == self.y_max-1 and j == 0:
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/3
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/3
						self.transitions[i, j, k, i, j] = (1 - self.probability)/3
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j+1] = self.transitions[i, j, k, i, j+1] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i-1, j]  + self.probability


					elif i == self.y_max-1 and j == self.y_max-1:
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/3
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/3
						self.transitions[i, j, k, i, j] = (1 - self.probability) /3
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i-1, j]  + self.probability
					elif i == 0:
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j] = (1 - self.probability)/4
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j+1] = self.transitions[i, j, k, i, j+1] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
					elif j == 0:
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/4
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j] = (1 - self.probability)/4
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j+1] = self.transitions[i, j, k, i, j+1] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i-1, j]  + self.probability
					elif i == self.y_max-1:
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j+1] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j] = (1 - self.probability)/4
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j+1] = self.transitions[i, j, k, i, j+1] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i-1, j]  + self.probability
					elif j == self.x_max-1:
						self.transitions[i, j, k, i+1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j-1] = (1 - self.probability)/4
						self.transitions[i, j, k, i-1, j] = (1 - self.probability)/4
						self.transitions[i, j, k, i, j] = (1 - self.probability)/4
						if k == 0:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j]  + self.probability
						elif k == 1:
							self.transitions[i, j, k, i, j] = self.transitions[i, j, k, i, j] + self.probability	
						elif k == 2:
							self.transitions[i, j, k, i+1, j] = self.transitions[i, j, k, i+1, j]  + self.probability	
						elif k == 3:
							self.transitions[i, j, k, i, j-1] = self.transitions[i, j, k, i, j-1]  + self.probability
						elif k == 4:
							self.transitions[i, j, k, i-1, j] = self.transitions[i, j, k, i-1, j]  + self.probability

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
					'''		
		if sink is True:
			for k in range(5):
				#self.transitions[self.loc_min_0[0], self.loc_min_0[1], k] = np.zeros([y_max, x_max])
				#self.transitions[self.loc_min_0[0], self.loc_min_0[1], k, self.loc_min_0[0], self.loc_min_0[1]] = 1.0
				self.transitions[self.loc_max_0[0], self.loc_max_0[1], k] = np.zeros([y_max, x_max])
				self.transitions[self.loc_max_0[0], self.loc_max_0[1], k, self.loc_max_0[0], self.loc_max_0[1]] = 1.0
				self.transitions[self.loc_max_1[0], self.loc_max_1[1], k] = np.zeros([y_max, x_max])
				self.transitions[self.loc_max_0[0], self.loc_max_0[1], k, self.loc_max_0[0], self.loc_max_0[1]] = 1.0

		

		

		file = open('state_space', 'w')
		file.write("x_max\n")
		file.write(str(self.x_max-1))
		file.write("\ny_max\n")
		file.write(str(self.y_max-1))
		file.write("\np\n")
		file.write(str(self.probability))
		file.write("\nx_h_0\n")
		file.write(str(self.loc_max_0[0]))
		file.write("\ny_h_0\n")
		file.write(str(self.loc_max_0[1]))
		file.write("\nx_h_1\n")
		file.write(str(self.loc_max_0[0]))
		file.write("\ny_h_1\n")
		file.write(str(self.loc_max_0[1]))
		file.write("\nx_l_0\n")
		file.write(str(self.loc_min_0[0]))
		file.write("\ny_l_0\n")
		file.write(str(self.loc_min_0[1]))
		file.write("\nx_l_1\n")
		file.write(str(self.loc_min_1[0]))
		file.write("\ny_l_1\n")
		file.write(str(self.loc_min_1[1]))

		file.close()


	#initialize a grid, all rewards are settled in each coord
	def w_features(self, theta):
		theta=theta/np.linalg.norm(theta)
		for i in range(self.y_max):
			for j in range(self.x_max):
				self.rewards[i, j]=np.dot(np.array(theta), np.transpose(self.features[i, j]))
	
		return self.rewards

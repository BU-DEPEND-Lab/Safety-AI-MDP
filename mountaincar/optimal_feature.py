from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab

def optimal_feature(grids, policy, epsilon = 1e-2, gamma= 0.99):
	exp_u= np.zeros(3)
	features= np.array(grids.features)
	diff = float("inf")
	while diff > epsilon:
		diff = 0.
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				features_temp = features
				action = policy[i, j]
				transition = np.array(grids.transitions[i, j, action])
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						features_temp[i, j] = features_temp[i, j] + np.multiply(transition[m, n], gamma * features[m, n])	
				new_diff = np.linalg.norm(features[i, j] - features_temp[i, j], ord= 1)
				if new_diff > diff:
					diff = new_diff
		features=features_temp
	for i in range(grids.y_max):
		for j in range(grids.x_max):
			exp_u = exp_u + features[i, j]
	exp_u = exp_u/(grids.x_max * grids.y_max)
	return exp_u



def optimal_value(grids, epsilon = 1e-2, gamma = 0.99):
	values = np.array(grids.rewards)
	diff = float("inf")
	while diff > epsilon:
		diff = 0.
		values_temp = values
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				max_value = float("-inf")
				for k in range(5):
					value_k= grids.rewards[i, j]
					transition_k = grids.transitions[i, j, k]
					reward_k = np.multiply(transition_k, gamma * values)
					for m in range(grids.y_max):
						for n in range(grids.x_max):
							value_k+= reward_k[m, n]
					max_value = max(value_k, max_value)		
				new_diff = abs(values[i, j] - max_value)
				if new_diff >  diff:
					diff = new_diff
				values_temp[i, j] = max_value
		values = values_temp
	return values	

def update_policy(grids, epsilon=0.01, gamma=0.99):	
	policy=np.ones([grids.y_max, grids.x_max])
	values = optimal_value(grids, epsilon=epsilon, gamma=gamma)
	print "new value function generated"
	print values
	Q = np.zeros([grids.x_max, grids.y_max, 5])
	for i in range(grids.y_max):
		for j in range(grids.x_max):
			for k in range(5):
				value_k= grids.rewards[i, j]
				transition_k = grids.transitions[i, j, k]
				reward_k = np.multiply(transition_k, gamma * values)
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						value_k+= reward_k[m, n]
				Q[i, j, k] = value_k
		##Q -= Q.max(axis=2).reshape((grids.y_max, grids.x_max, 1))
		##Q = np.exp(Q)/np.exp(Q).sum(axis=2)
			policy[i, j] = np.argmax(Q[i, j])
	return policy

def draw_grids(rewards, trajectory= None):
	c=pylab.pcolor(rewards)
	x=[]
	y=[]
	if trajectory!=None:
		for i in trajectory:
			x.append(i["state"][0])
			y.append(i["state"][1])
			pylab.plot(x, y, 'bo', x, y, 'b-', [x[-1]], [y[-1]], 'ro')
	c = pylab.pcolor(rewards, edgecolors='w', linewidths=1)
	pylab.set_cmap('gray')
	pylab.axis([0,len(rewards)-1, len(rewards)-1,0])
	pylab.savefig('plt.png')
	pylab.show()

if __name__=="__main__":
	grids=grid(10, 10)
	grids.w_features(np.array([1./3., 1./3., -1./3.]))
	print grids.rewards
	agent= car(states=grids.states)
	policy= update_policy(grids, 0.001, 0.5)
	print policy
	exp_u= optimal_feature(grids, policy, 1e-2, 0.5)
	print exp_u	
	draw_grids(grids.rewards)

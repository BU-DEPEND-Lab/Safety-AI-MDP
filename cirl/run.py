from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab
import sys
from update import train


if __name__ == "__main__":
	real=raw_input("Try synthesizing? [Y/N]")
	if real == 'Y' or real == 'y':
		grids= grid(8, 8, False, [[4, 3], [1, 2], [1, 1], [4, 4]], 0.7, sink = False)
		trains = train(grids = grids, gamma = 0.99, iteration = 20, starts = [grids.states[0][0]], steps = None, epsilon = 0.0001, theta = [ 1.0, 0.5, -1.0, 0.0])
		starts = [np.array([0, 0])]
		trains.synthesize(theta = [1.0, 0.5, -1.0, 0.0], starts = starts)

	real=raw_input("Learn from policy?[Y/N]")
	if real == 'Y' or real == 'y':
		grids= grid(8, 8, False, [[4, 3], [1, 2], [1, 1], [4, 4]], 0.7, sink = False)
		trains = train(grids = grids, gamma = 0.99, iteration = 20, starts = [grids.states[0][0]], steps = None, epsilon = 0.0001, theta = [ 1.0, 0.5, -1.0, 0.0])
		print "Start real expert training..."
		starts3 = [np.array([0, 0])]
		policy=trains.real_expert_train(starts3, distribution= None)
		
		print policy
		policy[grids.loc_min_0[0]-1, grids.loc_min_0[1]] = 1
		policy[grids.loc_min_0[0], grids.loc_min_0[1]-1] = 2
		policy[grids.loc_min_0[0]+1, grids.loc_min_0[1]] = 1
		#policy[grids.loc_min_0[0], grids.loc_min_0[1]+1] = 1
		policy[grids.loc_min_0[0]+1, grids.loc_min_0[1]+1] = 2
		#policy[grids.loc_min_0[0]-1, grids.loc_min_0[1]+1] = 1
		policy[grids.loc_min_0[0]+1, grids.loc_min_0[1]-1] = 2
		#policy[grids.loc_min_0[0]+2, grids.loc_min_0[1]] = 3
		'''
		policy= np.array([[1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] , 
				  [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0] , 
				  [1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0, 2.0] , 
     				  [1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 2.0, 2.0] , 
				  [1.0, 1.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0] , 
				  [1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0] , 
				  [1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0, 2.0] , 
				  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] , 
				])
		'''

		print "policy modified"	
		policy=trains.learn_from_policy(starts3, policy)
		print policy
		
		grids = None
		trains = None
		
	
	real=raw_input("Try real expert training with random initial state and random features? [Y/N]")
	if real == 'Y' or real == 'y':
		for i in range(100):
			print "experiment ", i
			grids= grid(8, 8, True, 0.7, sink = True)
			trains = train(grids = grids, gamma = 0.99, iteration = 20, starts = [grids.states[0][0]], steps = None, epsilon = 0.0001, theta = [1.0, 0.0, 0.0, 0.0])
			print "Start real expert training..."
			starts3 = [np.array([0, 0])]
			trains.real_expert_train(starts3, distribution=None)
			grids = None
			trains = None
		
	
	real=raw_input("Try human expert trainin? [Y/N]")
	if real == 'Y' or real == 'y':
		grids= grid(8, 8, False, [[7, 7], [3, 3], [5, 5], [4, 4]], 0.7, sink = False)
		trains = train(grids = grids, gamma = 0.99, iteration = 20, starts = [grids.states[0][0]], steps = None, epsilon = 0.0001, theta = [ 1.0, 0.5, -1.0, 0.0])
		policy_temp = trains.human_train()
	
	real=raw_input("Try real expert training with random optimal policy? [Y/N]")
	if real == 'Y' or real == 'y':
		distribution = []
		for i in range(11):
			distribution.append(1.0-i/10.0)
		grids= grid(8, 8, False, [[7, 7], [3, 3], [5, 5], [4, 4]], 0.7,sink = True)
		trains = train(grids = grids, gamma = 0.99, iteration = 20, starts = [grids.states[0][0]], steps = None, epsilon = 0.0001, theta = [1.0, 0.5, -1.0, 0.0])
		print "Start random expert optimal training..."
		starts3 = [np.array([0, 0])]
		trains.real_expert_train(starts3, distribution=distribution)


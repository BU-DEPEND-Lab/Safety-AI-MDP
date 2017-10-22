from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab


def optimal_value(grids, epsilon = 1e-2, gamma = 0.5):
	values = np.array(grids.rewards)
	diff = float("inf")
	while diff > epsilon:
		hehe=input("continue?")
		diff = 0.
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				max_value = float("-inf")
				value_i_j=np.zeros(5)
				for k in range(5):
					transition_k = grids.transitions[i, j, k]
					reward_k = np.multiply(transition_k, gamma * values)
					for m in range(grids.y_max):
						for n in range(grids.x_max):
							value_i_j[k]= value_i_j[k] + reward_k[m, n]
					value_i_j[k] = value_i_j[k] + grids.rewards[i, j]
					max_value = max(value_i_j[k], max_value)		
				new_diff = abs((values[i, j] - max_value))
				if new_diff >  diff:
					diff = new_diff
				values[i, j] = max_value
	return values	

if __name__=="__main__":
	grids=grid()
	grids.w_features([1./3., 1./3., 1./3.])
	print grids.rewards
	print(optimal_value(grids, 0.01))



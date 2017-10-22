from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab


def calc_u(grids, agent, policy, step, gamma=0.5):
	mu=np.zeros(3)
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0], agent.state[1]]}]
	for i in range(step):
		action=policy[agent.state[0], agent.state[1]]
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(action)})
		trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0], trajectory[-1]["state"][1]])
	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	return mu, trajectory



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
	agent=car(states=grids.states)
	policy=np.ones([grids.y_max, grids.x_max])
	mu, trajectory = calc_u(grids, agent, policy, 1000)
	print mu
	for i in trajectory:
		print i["action"]
		print i["state"]
		print i["feature"]

from car import car
from grid import grid

from cvxopt import matrix, solvers
import os
import numpy as np
import matplotlib
import pylab
import warnings
import random
import subprocess, shlex
from threading import Timer

warnings.filterwarnings("ignore")

def real_optimal(grids, agent, starts, steps, theta = None, gamma=0.5, epsilon = 1e-5):
	expert=[]
	if theta is None:
		theta = np.array([1./3., 1./3., -3./3., 0.0])
	theta = theta/np.linalg.norm(theta, ord=2)
	grids.w_features(theta)
	#optimal_policy= update_policy(grids, steps= steps, epsilon= epsilon, gamma= gamma)
	optimal_policy, _ = optimal_value(grids, steps= steps, epsilon= epsilon, gamma= gamma)
	print "real optimal policy generated"
	print "["
	for i in range(len(optimal_policy)):
		temp = []
		for j in range(len(optimal_policy[i])):
			temp.append(optimal_policy[i, j])
		print temp, ", "
	print "]" 
	#print optimal_policy
	file = open('optimal_policy', 'w')
	for i in optimal_policy:
		for j in i:
			file.write(str(j)+":")
		file.write("\n")
	file.close()

	opt_u = optimal_feature(grids, starts, steps, optimal_policy, epsilon, gamma)
	return optimal_policy, opt_u
	


def demo(grids, agent, start, steps, theta = None, gamma=0.5, epsilon = 1e-5):
	expert={}
	agent.state=np.array(grids.states[start[0], start[1]])
	if theta is  None:
		theta=np.array([1./3., 1./3., -3./3., 0.0])
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0]][agent.state[1]]}]
	grids.w_features(theta)
	pylab.close()
	pylab.ion()
	pylab.title("Generate demonstration[0:end, 1: left, 2: down, 3: right, 4: up]")
	draw_grids(grids.rewards, trajectory)
	print grids.rewards
	mu=np.zeros(4)
	action = 5
	while(steps > 0 and action != 0):
		try:
			action = input("%0.0f steps left, action is " % steps)
			if steps == float("inf") and action == 0:
				pylab.ioff()
				pylab.close('all')
				break
			steps = steps - 1	
			if action!= 0 and action != 1 and action !=2 and action !=3 and action !=4:
				print("Invalid action, input again")
				next
			else:
				trajectory[-1]["action"]=action
				trajectory.append({"state": agent.move(grids.transitions, action)})
				trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0]][trajectory[-1]["state"][1]])
				grids.w_features(theta)
				draw_grids(grids.rewards, trajectory)
		except:
			print("Invalid action, input again")
			next

	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	diff = float("inf")
	while diff > epsilon:
		i = i + 1
		diff =  (gamma**i) * trajectory[-1]["feature"]
		mu = mu + diff
		diff = np.linalg.norm(diff, ord = 2)
		
	expert["mu"]=mu
	expert["trajectory"]=trajectory
 	playagain=raw_input("Want to play again? [y/n]?")
	return expert, playagain
	
def calc_u(grids, agent, policy, steps, gamma=0.5):
	mu=np.zeros(3)
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0]][agent.state[1]]}]
	for i in range(steps):
		action=policy[agent.state[0], agent.state[1]]
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(grids.transitions, action)})
		trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0]][trajectory[-1]["state"][1]])
	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	return mu, trajectory

def exp_u(grids, agent, policy, start, start_action=None, steps=None, epoch=1000, gamma=0.5):
	if steps is None:
		steps = epoch
	mu=np.zeros([3])
	agent.state=np.array([start[0], start[1]])
	trajectory_i_j={}
	for i in range(epoch):
		if start_action is not None:
			org_action=policy[agent.state[0], agent.state[1]]
			policy[agent.state[0], agent.state[1]]=start_action
			mu_i_j_1, _= calc_u(grids, agent, policy , steps=1)
			policy[agent.state[0], agent.state[1]]=org_action
		else:
			mu_i_j_1 = np.zeros([3])
		mu_i_j, _ =calc_u(grids, agent, policy, steps=steps)
		mu = mu + mu_i_j + mu_i_j_1
		#	draw(rewards, trajectory)
	mu=mu/epoch
	return mu


def draw_grids(rewards, trajectory):
	pylab.set_cmap('gray')
	pylab.axis([0,len(rewards[0]), len(rewards),0])
	c = pylab.pcolor(rewards, edgecolors='w', linewidths=1)
	
	x=[]
	y=[]
	if trajectory!=None:
		for i in trajectory:
			y.append(i["state"][0])
			x.append(i["state"][1])
			pylab.plot(x, y, 'bo', x, y, 'b-', [x[-1]], [y[-1]], 'ro')
	pylab.show()

	


def sample_feature(grids, agent, starts, STEP, policy, epochs = 1000, epsilon = 1e-5, gamma = 0.5, bounds = None):
	u =  np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	u_G = np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	u_B = np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	p_B = np.zeros([len(grids.features), len(grids.features[-1])])
	p_G = np.zeros([len(grids.features), len(grids.features[-1])])
	B = np.zeros([len(grids.features), len(grids.features[-1]), 2])
	P = np.zeros([len(grids.features), len(grids.features[-1])])
	if STEP is None:
		STEP = grids.x_max * grids.y_max
	if bounds is None:
		bounds = []
		for start in range(len(starts)):
			bounds.append(STEP*2)

	for start in range(len(starts)):
		print starts[start]
		print bounds[start]
	 	epochs_G=0.0
	 	epochs_B=0.0
 	 	while epochs_G + epochs_B <= epochs-1:
			agent.state = np.array(starts[start])
			path = [np.array(agent.state)]
			u_epoch = grids.features[agent.state[0]][agent.state[1]] 
			steps = 0
			end = False
			fail = False
			while steps < STEP and end is False:
				steps = steps + 1
				action_epoch = int(policy[agent.state[0], agent.state[1]])
				if fail is not True:
					agent.state = np.array(agent.move(grids.transitions, action_epoch))
					path.append(np.array(agent.state))
				if steps <= bounds[start] and agent.state[0] ==  grids.loc_min_0[0] and agent.state[1] == grids.loc_min_0[1]:
					fail = True
				u_epoch = u_epoch + grids.features[agent.state[0]][agent.state[1]] * (gamma**steps)
				#if np.linalg.norm(grids.features[agent.state[0]][agent.state[1]] * (gamma**steps), ord=2) <= epsilon:
				if gamma**steps <= epsilon:
					end = True
			if fail is True:
				for state in path:
					B[state[0]][state[1]][0] = B[state[0]][state[1]][0] + 1
					B[state[0]][state[1]][1] = B[state[0]][state[1]][1] + 1
				u_B[starts[start][0], starts[start][1]] = u_B[starts[start][0], starts[start][1]]  + u_epoch
				epochs_B=epochs_B+1
			else:							
				for state in path:
					B[state[0]][state[1]][0] = B[state[0]][state[1]][0] + 1
				u_G[starts[start][0], starts[start][1]]  = u_G[starts[start][0], starts[start][1]]  + u_epoch
				epochs_G=epochs_G+1
			if np.linalg.norm((u_B[starts[start][0], starts[start][1]]+u_G[starts[start][0], starts[start][1]])/(epochs_G+epochs_B), ord=2) <= epsilon and  epochs_G+epochs_B>=epochs/2:
				break
	
		u_B[starts[start][0], starts[start][1]]  = u_B[starts[start][0], starts[start][1]] /epochs_B
		u_G[starts[start][0], starts[start][1]]  = u_G[starts[start][0], starts[start][1]] /epochs_G
		if epochs_B <= 0.0:
			u_B[starts[start][0], starts[start][1]] =np.zeros(len(grids.features[-1][-1]))
		if epochs_G <= 0.0:
			u_G[starts[start][0], starts[start][1]] =np.zeros(len(grids.features[-1][-1]))
		p_B[starts[start][0], starts[start][1]]  = float(epochs_B/(epochs_B+epochs_G))
  		p_G[starts[start][0], starts[start][1]]  = float(epochs_G/(epochs_B+epochs_G))
		
		
		for i in range(len(B)):
			for j in range(len(B[i])):
				if B[i, j, 0] > 0:
					P[i, j] = float(B[i, j, 1]/B[i, j, 0])
	return u_G, p_G, u_B, p_B, P





def optimal_feature(grids, starts, steps, policy, epsilon = 1e-5, gamma= 0.5):
	exp_u= np.zeros(len(grids.features[-1][-1]))
	features= np.array(grids.features)
	'''
	if steps + 1 != steps:
		features_temp = np.array(grids.features)
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				action = int(policy[i, j])
				transition = np.array(grids.transitions[i, j, action])
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						features_temp[i, j] = features_temp[i, j] + np.multiply(transition[m, n], gamma * features[m][n])	
		features= np.array(features_temp)
	'''
	diff = float("inf")
	while diff > epsilon:
		diff = 0.
		features_temp = np.array(grids.features)
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				action = int(policy[i, j])
				transition = np.array(grids.transitions[i, j, action])
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						features_temp[i, j] = features_temp[i, j] + np.multiply(transition[m, n], gamma * features[m][n])	
				new_diff = np.linalg.norm(features[i, j] - features_temp[i, j], ord= 2)
				if new_diff > diff:
					diff = new_diff
		features=features_temp
	
	for i in range(len(starts)):
		exp_u = exp_u + features[starts[i][0]][starts[i][1]]
	exp_u = exp_u/len(starts)
	return exp_u



def optimal_value(grids, steps, epsilon = 1e-5, gamma = 0.5):
	values = np.array(grids.rewards)
	policy = np.zeros([grids.y_max, grids.x_max])
	'''
	if steps + 1 != steps:
		values_temp = np.array(values)
		while(steps>0):
			for i in range(grids.y_max):
				for j in range(grids.x_max):
					max_value = float("-inf")
					for k in range(5):
						transition_k = grids.transitions[i, j, k]
						reward_k = np.multiply(transition_k, gamma * values)
						value_k = 0.
						for m in range(grids.y_max):
							for n in range(grids.x_max):
								value_k+= reward_k[m, n]
						max_value = max(value_k, max_value)		
					values_temp[i, j] = grids.rewards[i, j] + max_value
			values = np.array(values_temp)
			steps = steps - 1
		return values	
	'''
	diff = float("inf")
	while diff > epsilon:
		diff = 0.
		values_temp = np.zeros([grids.y_max, grids.x_max])
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				max_value = float("-inf")
				for k in range(5):
					transition_k = grids.transitions[i, j, k]
					reward_k = np.multiply(transition_k, values)
					value_k = 0.
					for m in range(grids.y_max):
						for n in range(grids.x_max):
							value_k+= reward_k[m, n]
					if max_value < value_k:
						policy[i, j] = k
						max_value = value_k
				values_temp[i, j] = grids.rewards[i, j] + gamma * max_value
				new_diff = abs(values[i, j] - values_temp[i, j])
				if new_diff >  diff:
					diff = new_diff
		#values = np.array(values + 0.1 * (values_temp - values))
		values = np.array(values_temp)	
	return policy, values

#def update_policy(grids, steps, epsilon= 1e-5, gamma=0.5):	
#	policy=np.ones([grids.y_max, grids.x_max])
#	policy, values = optimal_value(grids, steps= steps-1, epsilon=epsilon, gamma=gamma)
#	Q = np.zeros([grids.x_max, grids.y_max, 5])
#	for i in range(grids.y_max):
#		for j in range(grids.x_max):
#			for k in range(5):
#				value_k= grids.rewards[i, j]
#				transition_k = grids.transitions[i, j, k]
#				reward_k = np.multiply(transition_k, gamma * values)
#				for m in range(grids.y_max):
#					for n in range(grids.x_max):
#						value_k+= reward_k[m, n]
#				Q[i, j, k] = value_k
#			policy[i, j] = np.argmax(Q[i, j])
#	return policy

def update_policy(grids, values, epsilon= 1e-5, gamma=0.5):	
	policy=np.ones([grids.y_max, grids.x_max])
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
			policy[i, j] = np.argmax(Q[i, j])
	return policy




def expert_train(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = None):
	if start_theta is None:
		start_theta=np.random.randint(-100, 100, 4)
	new_theta=start_theta/np.linalg.norm(start_theta, ord=2)
	grids.w_features(new_theta)
	thetas = [new_theta]
	new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = [new_policy]
	values = [new_value]
	if MC is False:
		new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	else:
		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
		new_mu = np.sum(np.reshape(exp_u_G, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_G, [grids.y_max*grids.x_max, 1]), 0)
		## +   np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0) 

	print "Initial theta ", new_theta
	print "Initial expected features ", new_mu
	#print "Initial expected feature error ", np.linalg.norm(expert-new_mu, ord=2)
	mus = [new_mu]
	flag = float("inf")
	new_index = 0
	index = 0
	
	for i in range(iteration):
		new_index, new_theta, w_delta_mu = expert_update_theta(grids, expert, agent, steps, policies, mus, gamma, epsilon)
		new_theta = new_theta/np.linalg.norm(new_theta, ord=2)	
		print i, " iteration", "policy ", new_index, " weighted delta mu: ", w_delta_mu, "new theta: ", new_theta 
	
		print "start learning...."
		grids.w_features(new_theta)
		#if weighted weighted feature approximates the expert, end training
		new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
		print "new policy generated...begin next iteration"
		if MC is False:
			new_mu =  optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		else:
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
			new_mu = np.sum(np.reshape(exp_u_G, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_G, [grids.y_max*grids.x_max, 1]) +   np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0) 

		thetas.append(new_theta)
		policies.append(new_policy)
		values.append(new_value)
		mus.append(new_mu)
		print "new policy expected feature", new_mu
		print "new policy expected feature error ", np.linalg.norm(expert-new_mu, ord=2)
		if np.linalg.norm(expert-new_mu, ord=2) < np.linalg.norm(expert-mus[index], ord=2):
			index = len(mus)-1
			print "policy ", index, " is the new best learnt policy"
			if safety is not None:
				exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, policies[index], epochs= 5000, epsilon = epsilon, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
				print "best policy's unsafe rate ", p_B_sum
				if p_B_sum > safety:
					mus[index] = mus[index] -  (p_B_sum - safety) * np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0)
				print "feature counts modified to ", mus[index]

				
		if abs(w_delta_mu) < epsilon:
			print "|expert_w_mu - w_mu| = ", abs(w_delta_mu), " < ", epsilon
			#index = new_index
			break	

		if np.linalg.norm(mus[-1]-mus[-2], ord=2)<=epsilon and np.linalg.norm(thetas[-1]-thetas[-2], ord=2)<=epsilon:
			print "Difference with last iteration is too small"
			break	
	#index = -1
	#print "best policy"
	#print policies[index]
	print "best weight", thetas[index]
	print "best feature", mus[index]
	print "best policy\n", policies[index]
	grids.w_features(thetas[index])
	#draw_grids(grids.rewards, None)
	return grids, thetas[index], policies[index], values[index]


def expert_train_v1(grids, experts, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = None):
	if start_theta is None:
		start_theta=np.random.randint(-100, 100, 4)
	new_theta=start_theta/np.linalg.norm(start_theta, ord=2)
	expert = np.zeros(len(grids.features[-1][-1]))
	grids.w_features(new_theta)
	thetas = [new_theta]
	new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = [new_policy]
	values = [new_value]
	if MC is False:
		new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	else:
		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
		new_mu = np.sum(np.reshape(exp_u_G, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_G, [grids.y_max*grids.x_max, 1]), 0)
		## +   np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0) 

	print "Initial theta ", new_theta
	print "Initial expected features ", new_mu
	print "Initial expected feature error ", np.linalg.norm(expert-new_mu, ord=2)
	mus = [new_mu]
	flag = float("inf")
	new_index = 0
	index = 0
	
	for i in range(iteration):
		new_index, new_theta, w_delta_mu = expert_update_theta_v1(grids, experts, agent, steps, policies, mus, gamma, epsilon)
		new_theta = new_theta/np.linalg.norm(new_theta, ord=2)	
		print i, " iteration", "[CEX, expert, policy] = ", new_index, " weighted delta mu: ", w_delta_mu, "new theta: ", new_theta 
		
		expert = np.zeros(len(grids.features[-1][-1]))
		for i in range(len(new_index)-1):
			expert = expert + experts[0][i] * experts[1][i][new_index[i]]
		print "Combinatorial expert feature ", expert
	
		print "start learning...."
		grids.w_features(new_theta)
		#if weighted weighted feature approximates the expert, end training
		new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
		print "new policy generated...begin next iteration"
		if MC is False:
			new_mu =  optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		else:
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
			new_mu = np.sum(np.reshape(exp_u_G, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_G, [grids.y_max*grids.x_max, 1]) +   np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0) 

		thetas.append(new_theta)
		policies.append(new_policy)
		values.append(new_value)
		mus.append(new_mu)
		print "new policy expected feature", new_mu
		print "new policy expected feature error ", np.linalg.norm(expert-new_mu, ord=2)
		if np.linalg.norm(expert-new_mu, ord=2) < flag:
			index = len(mus)-1
			flag = np.linalg.norm(expert-new_mu, ord=2)
			print "policy ", index, " is the new best learnt policy"
			if safety is not None:
				exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, policies[index], epochs= 5000, epsilon = epsilon, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
				print "best policy's unsafe rate ", p_B_sum
				if p_B_sum > safety:
					mus[index] = mus[index] -  (p_B_sum - safety) * np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0)
				print "feature counts modified to ", mus[index]

		else:
			print "policy ", index, " is still the best learnt policy"	
		if abs(w_delta_mu) < epsilon:
			print "|expert_w_mu - w_mu| = ", abs(w_delta_mu), " < ", epsilon
			index = new_index[-1]
			print "policy ", index, " is the new best learnt policy"
			break	
		
		if np.linalg.norm(mus[-1]-mus[-2], ord=2)<=epsilon and np.linalg.norm(thetas[-1]-thetas[-2], ord=2)<=epsilon:
			print "Difference with last iteration is too small"
			print "policy ", index, " is the new best learnt policy"
			break	
	#index = -1
	#print "best policy"
	#print policies[index]
	print "best weight", thetas[index]
	print "best feature", mus[index]
	print "best policy\n", policies[index]	
	grids.w_features(thetas[index])
	#draw_grids(grids.rewards, None)
	return grids, thetas[index], policies[index], values[index]
def expert_update_theta(grids, expert, agent, steps, policies, mus, gamma=0.5, epsilon = 1e-5):
	#mus=[]
	delta_mus = []
	w_delta_mus=[]
	solutions=[]
	exp_mu = expert
	#for policy in policies:
	#	mu = optimal_feature(grids, steps, policy, epsilon = epsilon, gamma=gamma)
	#	mus.append(mu)
	
	for i in range(len(mus)):
		#G_i=[[], [], [], []]
		#h_i = []
		G_i = [[- (exp_mu[0] - mus[i][0])],
			 [- (exp_mu[1] - mus[i][1])],
			 [- (exp_mu[2] - mus[i][2])],
			 [- (exp_mu[3] - mus[i][3])]
			]
		h_i = [0]
		c = matrix(mus[i] - exp_mu)
		for j in range(len(mus)):
			G_i[0].append(-1 * mus[i][0] - (-1) * mus[j][0])
			G_i[1].append(-1 * mus[i][1] - (-1) * mus[j][1])
			G_i[2].append(-1 * mus[i][2] - (-1) * mus[j][2])
			G_i[3].append(-1 * mus[i][3] - (-1) * mus[j][3])
			h_i.append(0)

		G_i[0]= G_i[0] + [0., -1., 0., 0., 0.]
		G_i[1]= G_i[1] + [0., 0., -1., 0., 0.]
		G_i[2]= G_i[2] + [0., 0., 0., -1., 0.]
		G_i[3]= G_i[3] + [0., 0., 0., 0., -1.]
		h_i = h_i + [1., 0., 0., 0., 0.]

		G = matrix(G_i)
	#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
		h = matrix(h_i)

		dims = {'l': 1 + len(mus), 'q': [5], 's': []}
		sol = solvers.conelp(c, G, h, dims)
		sol['status']
		solution = np.array(sol['x'])
		if solution is not None:
			solution=solution.reshape(4)
			w_delta_mu=np.dot(solution, exp_mu - mus[i])
			w_delta_mus.append(w_delta_mu)
		else:
			w_delta_mus.append(None)
		solutions.append(solution)
	index = np.argmax(w_delta_mus)
	
	#solution = delta_mus[index]/np.linalg.norm(delta_mus[index], ord =2)
	#delta_mu = np.linalg.norm(delta_mus[index], ord =2)  
	return index, solutions[index], w_delta_mus[index]

def expert_update_theta_v1(grids, experts, agent, steps, policies, mus, gamma=0.5, epsilon = 1e-5, safety = None):
	mu = np.zeros(len(grids.features[-1][-1]))
	mu_B = np.zeros(len(grids.features[-1][-1]))

	mu_Bs = []
	for i in experts[1][0]:
		mu_Bs.append(i)
	
	delta_mus = []
	w_delta_mus=np.zeros([len(mus), len(mu_Bs)])
	solutions=np.zeros([len(mus), len(mu_Bs), 4])
	exp_mu = experts[-1][-1][-1]
	indices = []
	index = [0, 0, 0]
	safety = experts[0][0]
	max_w_delta_mus = -float('inf')
	for i in range(len(mus)):
		for j in range(len(mu_Bs)):
			G_i_j=[[], [], [], []]
			h_i_j = []
			c = matrix(- ((1 - safety) * (exp_mu  -  mus[i]) + safety * (- mu_Bs[j])))

			#G_i_j = [[- (exp_mu[0] - mus[i][0])],
			#	 [- (exp_mu[1] - mus[i][1])],
			#	 [- (exp_mu[2] - mus[i][2])],
			#	 [- (exp_mu[3] - mus[i][3])]
			#	]
			#h_i_j = [0]

			#G_i_j[0].append(- (mus[j][0] - mu_Bs[i][0]))
			#G_i_j[1].append(- (mus[j][1] - mu_Bs[i][0]))
			#G_i_j[2].append(- (mus[j][2] - mu_Bs[i][0]))
			#G_i_j[3].append(- (mus[j][3] - mu_Bs[i][0]))
			#h_i_j.append(0)
			
						
			#G_i_j[0].append(- ((1 - safety) * (exp_mu[0] - mus[i][0]) + safety * (mus[i][0] - mu_Bs[j][0])))
			#G_i_j[1].append(- ((1 - safety) * (exp_mu[1] - mus[i][1]) + safety * (mus[i][1] - mu_Bs[j][1])))
			#G_i_j[2].append(- ((1 - safety) * (exp_mu[2] - mus[i][2]) + safety * (mus[i][2] - mu_Bs[j][2])))
			#G_i_j[3].append(- ((1 - safety) * (exp_mu[3] - mus[i][3]) + safety * (mus[i][3] - mu_Bs[j][3])))
			#h_i_j.append(0)

			for m in range(len(mus)):
				for n in range(len(mu_Bs)):
					G_i_j[0].append(- safety * mu_Bs[j][0] - (1 - safety) * mus[i][0] - ( - (1 - safety) * mus[m][0] - safety * mu_Bs[n][0]))
					G_i_j[1].append(- safety * mu_Bs[j][1] - (1 - safety) * mus[i][1] - ( - (1 - safety) * mus[m][1] - safety * mu_Bs[n][1]))
					G_i_j[2].append(- safety * mu_Bs[j][2] - (1 - safety) * mus[i][2] - ( - (1 - safety) * mus[m][2] - safety * mu_Bs[n][2]))
					G_i_j[3].append(- safety * mu_Bs[j][3] - (1 - safety) * mus[i][3] - ( - (1 - safety) * mus[m][3] - safety * mu_Bs[n][3]))
					h_i_j.append(0)

					#G_i_j[0].append(- mu_Bs[j][0] - (- mu_Bs[n][0]))
					#G_i_j[1].append(- mu_Bs[j][1] - (- mu_Bs[n][1]))
					#G_i_j[2].append(- mu_Bs[j][2] - (- mu_Bs[n][2]))
					#G_i_j[3].append(- mu_Bs[j][3] - (- mu_Bs[n][3]))
					#h_i_j.append(0)
		
				G_i_j[0].append(- (exp_mu[0] - mus[m][0]))
				G_i_j[1].append(- (exp_mu[1] - mus[m][1]))
				G_i_j[2].append(- (exp_mu[2] - mus[m][2]))
				G_i_j[3].append(- (exp_mu[3] - mus[m][3]))
				h_i_j.append(0)
					
			for n in range(len(mu_Bs)):
				G_i_j[0].append(- mu_Bs[j][0] - (- mu_Bs[n][0]))
				G_i_j[1].append(- mu_Bs[j][1] - (- mu_Bs[n][1]))
				G_i_j[2].append(- mu_Bs[j][2] - (- mu_Bs[n][2]))
				G_i_j[3].append(- mu_Bs[j][3] - (- mu_Bs[n][3]))
				h_i_j.append(0)
				

			G_i_j[0]= G_i_j[0] + [0., -1., 0., 0., 0.]
			G_i_j[1]= G_i_j[1] + [0., 0., -1., 0., 0.]
			G_i_j[2]= G_i_j[2] + [0., 0., 0., -1., 0.]
			G_i_j[3]= G_i_j[3] + [0., 0., 0., 0., -1.]
			h_i_j = h_i_j + [1., 0., 0., 0., 0.]

			G = matrix(G_i_j)
		#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
			h = matrix(h_i_j)
			dims = {'l':  len(mus) + len(mu_Bs) + len(mu_Bs) * len(mus), 'q': [5], 's': []}
			sol = solvers.conelp(c, G, h, dims)
			sol['status']
			solution = np.array(sol['x'])
			if solution is not None:
				solution=solution.reshape(4)
				w_delta_mu=np.dot(solution, (1 - safety) * (exp_mu - mus[i]) + safety * (- mu_Bs[j]))
				w_delta_mus[i][j] = w_delta_mu
			else:
				w_delta_mus[i][j] = None
			solutions[i][j]=solution
		indices.append(np.argmax(w_delta_mus[i]))
		if w_delta_mus[i][indices[-1]] > max_w_delta_mus:
			max_w_delta_mus = w_delta_mus[i][indices[-1]]
			index = [indices[-1], 0, i]
	#solution = delta_mus[index]/np.linalg.norm(delta_mus[index], ord =2)
	#delta_mu = np.linalg.norm(delta_mus[index], ord =2)  
	return index, solutions[index[-1]][index[0]], w_delta_mus[index[-1]][index[0]]


def expert_update_theta_v2(grids, expert, agent, steps, policies, mus, mu_Bs, gamma=0.5, epsilon = 1e-5, safety = None):
	if safety is None:
		safety = 0.5
	mu = np.zeros(len(grids.features[-1][-1]))
	mu_B = np.zeros(len(grids.features[-1][-1]))

	for i in range(len(mus)):
		mu =  mu + mus[i]
	mu = mu/len(mus)
	p = len(mu)/(len(mu) + len(mus))
	for j in range(len(mu_Bs)):
		mu_B = mu_B + mu_Bs[j]
	mu_B = mu_B/len(mu_Bs)

	delta_mus = []
	w_delta_mus=np.zeros([len(mus), len(mu_Bs)])
	solutions=np.zeros([len(mus), len(mu_Bs), 4])
	exp_mu = expert
	indices = []
	index = (0, 0)
	#for policy in policies:
	#	mu = optimal_feature(grids, steps, policy, epsilon = epsilon, gamma=gamma)
	#	mus.append(mu)
	max_w_delta_mus = 0.0
	safety = len(mu_Bs)
	for i in range(len(mus)):
		for j in range(len(mu_Bs)):
			G_i_j=[[], [], [], []]
			h_i_j = []
			c = matrix(- ((exp_mu- mus[i]) - (exp_mu - mu)) * p + (1-p) * (- mu_Bs[j] - (- mu_B)))

			#G_i_j = [[- (exp_mu[0] - mus[i][0])],
			#	 [- (exp_mu[1] - mus[i][1])],
			#	 [- (exp_mu[2] - mus[i][2])],
			#	 [- (exp_mu[3] - mus[i][3])]
			#	]
			#h_i_j = [0]

			G_i_j[0].append(mu_Bs[j][0])
			G_i_j[1].append(mu_Bs[j][1])
			G_i_j[2].append(mu_Bs[j][2])
			G_i_j[3].append(mu_Bs[j][3])
			h_i_j.append(0)
			
						
			#G_i_j[0].append(- (exp_mu[0] - mus[i][0] - safety * mu_Bs[j][0]))
			#G_i_j[1].append(- (exp_mu[1] - mus[i][1] - safety * mu_Bs[j][1]))
			#G_i_j[2].append(- (exp_mu[2] - mus[i][2] - safety * mu_Bs[j][2]))
			#G_i_j[3].append(- (exp_mu[3] - mus[i][3] - safety * mu_Bs[j][3]))
			#h_i_j.append(0)

			for m in range(len(mus)):
				for n in range(len(mu_Bs)):
					G_i_j[0].append((- mu_Bs[j][0] - mus[i][0]) * p - (1-p) * (- mus[m][0] - mu_Bs[n][0]))
					G_i_j[1].append((- mu_Bs[j][1] - mus[i][1]) * p - (1-p) * (- mus[m][1] - mu_Bs[n][1]))
					G_i_j[2].append((- mu_Bs[j][2] - mus[i][2]) * p - (1-p) * (- mus[m][2] - mu_Bs[n][2]))
					G_i_j[3].append((- mu_Bs[j][3] - mus[i][3]) * p - (1-p) * (- mus[m][3] - mu_Bs[n][3]))
					h_i_j.append(0)
		
				G_i_j[0].append(- mus[i][0] - (- mus[m][0]))
				G_i_j[1].append(- mus[i][1] - (- mus[m][1]))
				G_i_j[2].append(- mus[i][2] - (- mus[m][2]))
				G_i_j[3].append(- mus[i][3] - (- mus[m][3]))
				h_i_j.append(0)
					
			for n in range(len(mu_Bs)):
				G_i_j[0].append(- mu_Bs[n][0] - (- mu_Bs[j][0]))
				G_i_j[1].append(- mu_Bs[n][1] - (- mu_Bs[j][1]))
				G_i_j[2].append(- mu_Bs[n][2] - (- mu_Bs[j][2]))
				G_i_j[3].append(- mu_Bs[n][3] - (- mu_Bs[j][3]))
				h_i_j.append(0)
				

			G_i_j[0]= G_i_j[0] + [0., -1., 0., 0., 0.]
			G_i_j[1]= G_i_j[1] + [0., 0., -1., 0., 0.]
			G_i_j[2]= G_i_j[2] + [0., 0., 0., -1., 0.]
			G_i_j[3]= G_i_j[3] + [0., 0., 0., 0., -1.]
			h_i_j = h_i_j + [1., 0., 0., 0., 0.]

			G = matrix(G_i_j)
		#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
			h = matrix(h_i_j)
			dims = {'l':  1 + len(mu_Bs) + len(mus) + len(mu_Bs) * len(mus), 'q': [5], 's': []}
			sol = solvers.conelp(c, G, h, dims)
			sol['status']
			solution = np.array(sol['x'])
			if solution is not None:
				solution=solution.reshape(4)
				w_delta_mu=np.dot(solution, p * ((exp_mu- mus[i]) - (exp_mu - mu)) + (1-p) * (- mu_Bs[j] - (- mu_B)))
				w_delta_mus[i][j] = w_delta_mu
			else:
				w_delta_mus[i][j] = None
			solutions[i][j]=solution
		indices.append(np.argmax(w_delta_mus[i]))
		if w_delta_mus[i][indices[-1]] > max_w_delta_mus:
			max_w_delta_mus = w_delta_mus[i][indices[-1]]
			index = (i, indices[-1])
	#solution = delta_mus[index]/np.linalg.norm(delta_mus[index], ord =2)
	#delta_mu = np.linalg.norm(delta_mus[index], ord =2)  
	return index, solutions[index[0]][index[1]], w_delta_mus[index[0]][index[1]]



def multi_learn(grids, agent, theta, exp_policy, exp_mu, starts=None, steps=float("inf"), epsilon=1e-4, iteration=20, gamma=0.9, safety = 0.02):
	print "starting multiple goal learning"
	new_theta = theta
	new_policy = exp_policy
	new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "first theta is ", new_theta
	print "first feature is ", new_mu
	exp_u_G, p_G, exp_u_B, p_B, P = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = epsilon, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "when feature matched, unsafe rate ", p_B_sum
	while p_B_sum > safety:
		new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max * grids.x_max, len(grids.features[-1][-1])]), 0)
		
		print "failure path feature", new_mu_B
		new_mu = (new_mu - 0.5 * new_mu_B)/(1 - 0.5)	
		print "updated feature ", new_mu
		_, new_theta, new_policy, _= expert_train(grids, new_mu, agent, starts = starts, steps=steps, epsilon=epsilon, iteration=iteration, gamma=gamma, start_theta= None, MC = False, safety = None)
		print "new theta ", new_theta
		#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		print "new feature ", new_mu
		exp_u_G, p_G, exp_u_B, p_B, P = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = epsilon, gamma=gamma)
		p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		print "when feature matched, unsafe rate ", p_B_sum
	print "Finally safe"
	print "theta is ", new_theta	
	print "policy is "
	print new_policy 
	return new_policy


def expert_synthesize(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = 0.0001):
	print "Human demo feature ", expert
	print "Initial theta ", start_theta
	theta = start_theta
	flag = float('inf')
	index = []
	mu_Bs = []
	mus = []
	policies = []	
	grids.w_features(theta)
	new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = policies + [new_policy]
	start_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	#new_mu = start_mu
	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max * grids.x_max, 1]))
	print "Initial unsafe path rate ", p_B_sum
	if p_B_sum <= safety:
		return start_theta, new_policy, new_value, start_mu
	while p_B_sum > safety:
		new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
		mu_Bs = mu_Bs + [new_mu_B]
		print "Add counterexample features ", new_mu_B	

		print "Keep generating counterexamples until find a safe candidate" 
		new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
		#new_index, new_theta, w_delta_mu = expert_update_theta(grids, new_mu, agent, steps, policies, mu_Bs, gamma, epsilon)
		grids.w_features(new_theta)
		print "Weight learnt from counterexamples ", new_theta

		new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
		new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
		p_B_sum = np.sum(np.reshape(p_B, [grids.y_max * grids.x_max, 1]))
		print "Policy unsafe rate ", p_B_sum
     	print "Found 1st safe policy towards safety ", new_theta	
	#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	mus = mus + [new_mu]
	print "Corresponding feature ", new_mu
	flag = [new_theta, new_policy, new_value, new_mu]
	step = abs(np.linalg.norm(expert - new_mu, ord=2) - np.linalg.norm(expert - start_mu, ord=2))/(np.linalg.norm(expert - new_mu, ord=2) + np.linalg.norm(expert- start_mu, ord=2))
	print "Add weight towards safety with stepsize ", step
	temp_theta = start_theta + step * new_theta
	theta = temp_theta/np.linalg.norm(temp_theta, ord=2)
	grids.w_features(theta)
	print "New candidate weight ", theta
	new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
	policies = policies + [new_policy]
	mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "New candidate feature ", mu
	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	
	i = 1	
	while True:
		print ">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", theta, "\nunsafe prob:", p_B_sum, "\nfeature ", mu, "\ncurrent best deviation from expert: ", np.linalg.norm(expert - flag[-1], ord=2) 
		i = i + 1
		if p_B_sum > safety:
			while p_B_sum > safety:
				new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
				mu_Bs = mu_Bs + [new_mu_B]
				print "Add counterexample feature ", new_mu_B	

				print "Keep generating counterexamples until find a safe candidate" 
				new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
				grids.w_features(new_theta)
				print "Weight learnt from counterexamples ", new_theta

				new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
				exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max * grids.x_max, 1]))
				print "Policy unsafe rate ", p_B_sum
			print "Found safe weight towards safety ", new_theta
			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			mus = mus + [new_mu]
			print "Corresponding feature ", new_mu

			step = abs(np.linalg.norm(expert - new_mu, ord=2) - np.linalg.norm(expert - mu, ord=2))/(np.linalg.norm(expert - new_mu, ord=2) + np.linalg.norm(expert- mu, ord=2))
			print "Add weight towards safety with stepsize ", step
			temp_theta = theta + step * new_theta
			theta = temp_theta/np.linalg.norm(temp_theta, ord=2)
			grids.w_features(theta)
			print "New candidate weight ", theta

			new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
			policies = policies + [new_policy]
			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "New candidate feature ", mu
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		else:	
			if len(flag) > 0 and np.linalg.norm(flag[-1] - mu, ord=2) < epsilon:
				print "new safe theta ", theta, " \nfeature ", mu, "\ndifference with the best one", np.linalg.norm(flag[-1] - mu, ord=2), " unsafe rate ", p_B_sum
				flag = [theta, new_policy, new_value, mu]
				return grids, index[0], index[1], index[2]
			elif np.linalg.norm(expert - mu, ord=2)  < np.linalg.norm(expert - flag[-1], ord=2):
				flag = [theta, new_policy, new_value, mu]
				print "New best candidate found ", flag[0]

			print "Add new candidate policy expected feature", mu
			mus = mus + [mu]
			new_index, new_theta, w_delta_mu = expert_update_theta(grids, expert, agent, steps, policies, mus, gamma, epsilon)
			new_theta = new_theta/np.linalg.norm(new_theta, ord=2)
			grids.w_features(new_theta)
			print "Found weight towards human demo ", new_theta

			new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
			policies = policies + [new_policy]
			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", new_mu 
			'''
			new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
			policies = policies + [new_policy]
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			if p_B_sum > safety:
				new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
				mu_Bs = mu_Bs + new_mu_B
			else:
				new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
				mus = mus + new_mu
			'''
			step = abs((mu - new_mu)/(new_mu + mu))
			print "Add weight towards human demo with stepsize ", step
			temp_theta = theta + step * new_theta
			theta = temp_theta/np.linalg.norm(temp_theta, ord=2)
			grids.w_features(theta)
			print "New candidate weight ", theta

			new_policy, new_value  = optimal_value(grids, steps = steps, epsilon= epsilon, gamma = gamma)
			policies = policies + [new_policy]
			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "New candidate feature ", mu
			print "Deviate from expert ", np.linalg.norm(expert - mu, ord=2)
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, steps, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))			
	print "Iteration ended, best safe theta ", index[0]
	return grids, flag[0], flag[1], flag[2]



def expert_synthesize1(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = 0.0001):
	print "Human demo feature ", expert
	print "Initial theta ", start_theta
	flag = []
	index = []
	mu_Bs = []
	mus = []
	policies = []	
	grids.w_features(start_theta)
	start_policy, start_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = policies + [start_policy]
	start_mu = optimal_feature(grids, starts, steps, start_policy, epsilon = epsilon, gamma=gamma)
	expert = start_mu
	#new_mu = start_mu
	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, start_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "Initial unsafe path rate ", p_B_sum
	if p_B_sum <= safety:
		return start_theta, start_policy, start_value, start_mu
	while p_B_sum > safety:
		new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
		mu_Bs = mu_Bs + [new_mu_B]
		print "Add counterexample features ", new_mu_B	

		print "Keep generating counterexamples until find a safe candidate" 
		new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
		#new_index, new_theta, w_delta_mu = expert_update_theta(grids, new_mu, agent, steps, policies, mu_Bs, gamma, epsilon)
		grids.w_features(new_theta)
		print "Weight learnt from counterexamples ", new_theta

		new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
		new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
		p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		print "Policy unsafe rate ", p_B_sum
     	print "Found 1st safe policy towards safety ", new_theta	
	#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	mus = mus + [new_mu]
	print "Corresponding feature ", new_mu
	K = 1.0
	kk = 0.0
	k = K
	mu = k * new_mu + (1 - k) * expert
	print "1st combined feature ", mu
	grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, grids.x_max * grids.y_max, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
	policies = policies + [new_policy]
	grids.w_features(theta)
	print "Weight learnt from 1st combined feature ", theta

	mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "Corresponding feature ", mu

	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "Policy unsafe rate ", p_B_sum

	i = 1	
	while True:
		print ">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", theta, "\nunsafe prob:", p_B_sum, "\nfeature ", mu, " = ", k, "*safe + ", 1 - k, "*expert\n" 
		i = i + 1
		if p_B_sum > safety:
			print "Unsafe, learning from counterexample"
			if k > kk:
				kk = k
			while p_B_sum > safety:
				new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
				mu_Bs = mu_Bs + [new_mu_B]
				print "Add counterexample feature ", new_mu_B	

				print "Keep generating counterexamples until find a safe candidate" 
				new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
				grids.w_features(new_theta)
				print "Weight learnt from counterexamples ", new_theta

				new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
				exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
				print "Policy unsafe rate ", p_B_sum
			print "Found safe weight towards safety ", new_theta
			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			mus = mus + [new_mu]
			print "Corresponding feature ", new_mu
			
			k = k + (K - k)/2			
			mu = k * new_mu + (1 - k) * expert
			print "combined feature towards safety", mu
			grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			grids.w_features(theta)
			print "Weight learnt from combined feature ", theta
			policies = policies + [new_policy]

			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", mu

			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum
		else:	
			if len(flag) > 0 and abs(K - k) <= epsilon:
				print "Difference with the best one is too small", np.linalg.norm(flag[-1] - mu, ord=2)
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
				flag = [theta, new_policy, new_value, mu]
				return grids, flag[0], flag[1], flag[2]
			elif len(flag) > 0 and np.linalg.norm(expert - mu, ord=2)  <= np.linalg.norm(expert - flag[-1], ord=2):
			#elif len(flag) > 0 and np.dot(theta, expert) >= np.dot(flag[0], expert):
				flag = [theta, new_policy, new_value, mu]
				print "New best candidate"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
				K = k
				k = (K + kk)/2
			elif len(flag) == 0:
				flag = [theta, new_policy, new_value, mu]
				print "1st best candidate"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
				K = k
				k = (K + kk)/2

			else:
				print "Not the best"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
				k = (k + K)/2

			print "Add new candidate policy expected feature", mu
			mus = mus + [mu]
			
			mu = k * new_mu + (1 - k) * expert
			print "combined feature towards expert", mu
			grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			grids.w_features(theta)
			print "Weight learnt from combined feature ", theta
			policies = policies + [new_policy]

			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", mu
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum

	print "Iteration ended, best safe theta ", index[0]
	return grids, flag[0], flag[1], flag[2]


def expert_synthesize2(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = 0.0001):
	print "Human demo feature ", expert
	print "Initial theta ", start_theta
	flag = None
	index = []
	mu_Bs = []
	mus = []
	policies = []	
	grids.w_features(start_theta)
	start_policy, start_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = policies + [start_policy]
	start_mu = optimal_feature(grids, starts, steps, start_policy, epsilon = epsilon, gamma=gamma)
	expert = start_mu
	new_theta = np.array(start_theta)
	print "Model check ", start_theta
	p_B_sum = output_model(grids, starts, start_policy, steps, safety)
	#new_mu = start_mu
	#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, start_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "Initial unsafe path rate ", p_B_sum

	print "model output finished for initial policy"
	if p_B_sum <= safety and p_B_sum is not None:
		return start_theta, start_policy, start_value, start_mu
	elif p_B_sum is None:
		print "Model checking failed"
		return None
	while p_B_sum > safety:
		new_mu_B =  counterexample(grids, gamma, safety, epsilon)
		#new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
		if new_mu_B is not None:
			mu_Bs = mu_Bs + [new_mu_B]
			print "Add counterexample features ", new_mu_B	

		new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
		#new_index, new_theta, w_delta_mu = expert_update_theta(grids, new_mu, agent, steps, policies, mu_Bs, gamma, epsilon)
		grids.w_features(new_theta)
		print "Weight learnt from counterexamples ", new_theta

		new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
		print new_policy
		#print new_policy
		print "Model check ", new_theta
		p_B_sum = output_model(grids, starts, new_policy, steps, safety)
		#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
		#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		print "Policy unsafe rate ", p_B_sum
		print "Keep generating counterexamples until find a safe candidate" 
     	print "Found 1st safe policy towards safety ", new_theta	
	new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	mus = mus + [new_mu]
	print "Corresponding feature ", new_mu

	K = 1.0
	KK = 0.0
	k = 1.0
	mu = [[k,  1 - k],[mu_Bs, [expert]]]

		
	new_index, new_theta, w_delta_mu = expert_update_theta_v1(grids, mu, agent, steps, policies, mus, gamma, epsilon)
	#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
	grids.w_features(new_theta)
	print "Weight learnt from 1st combined feature ", new_theta
	new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "Corresponding feature ", new_mu
	#mus = mus + [new_mu]

	#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	#print "Policy unsafe rate ", p_B_sum
	print "Model check ", new_theta
	p_B_sum = output_model(grids, starts, new_policy, steps, safety)

	i = 1	
	while True:
		print ">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", new_theta, "\nunsafe probability:", p_B_sum, "\nfeature ", new_mu, " = ", k, "*safe + ", 1 - k, "*expert\n" 
		file = open('log', 'a')
		file.write(">>>>>>>>> " + str(i) + "th iteration, candidate theta: "+ str(new_theta) + "; unsafe probability: " + str(p_B_sum) + "; feature " + str(new_mu) + " = " + str(k) + "*safe + " + str(1 - k) + "*expert\n") 
		file.close()
		#file = open('log', 'a')
		#file.write(">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", new_theta, "\nunsafe prob:", p_B_sum, "\nfeature ", new_mu, " = ", k, "*safe + ", 1 - k, "*expert\n") 
		#file.close()
		i = i + 1
		
		if flag is not None and abs(K - k) < epsilon:
			print "Difference with the best one is too small", np.linalg.norm(flag['feature'] - new_mu, ord=2)
			#print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
			#print "Expert can get value ", np.dot(theta, expertw)
			#flag = [new_theta, new_policy, new_value, new_mu]
			#return grids, flag[0], flag[1], flag[2]
			break
		if p_B_sum is not None and p_B_sum > safety:
			print "Unsafe, learning from counterexample"
			#while p_B_sum > safety:
			#new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
			new_mu_B =  counterexample(grids, gamma, safety, epsilon)
			if new_mu_B is not None:
				mu_Bs = mu_Bs + [new_mu_B]
				print "Add counterexample features ", new_mu_B	

			#print "Keep generating counterexamples until find a safe candidate" 
			#new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
			#grids.w_features(new_theta)
			#print "Weight learnt from counterexamples ", new_theta

			#new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			#print "Policy unsafe rate ", p_B_sum
			#print "Found safe weight towards safety ", new_theta
			#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			#mus = mus + [new_mu]
			KK = k
			k = k + (K - k)/2			
			mu = [[k,  1 - k],[mu_Bs, [expert]]]
			#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			new_index, new_theta, w_delta_mu = expert_update_theta_v1(grids, mu, agent, steps, policies, mus, gamma, epsilon)
			grids.w_features(new_theta)
			print "Weight learnt from combined feature ", new_theta

			new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)

			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", new_mu
			#mus = mus + [new_mu]
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Model check ", new_theta
			p_B_sum = output_model(grids, starts, new_policy, steps, safety)
			print "Policy unsafe rate ", p_B_sum
			#mus = mus + [new_mu]
		elif p_B_sum is not None:	
			print "Safe, learning from expert"
			mus = mus + [new_mu]
			if flag is not None and np.linalg.norm(expert - new_mu, ord=2)  < np.linalg.norm(expert - flag['feature'], ord=2):
			#elif len(flag) > 0 and np.dot(theta, expert) > np.dot(flag[0], expert):
				flag = {'weight': new_theta, 'policy': new_policy, 'value': new_value, 'unsafe': p_B_sum, 'feature': new_mu}
				print "New best candidate"
				print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(theta, expert)
				K = k
				k = (K + KK)/2
			elif flag is None:
				flag = {'weight': new_theta, 'policy': new_policy, 'value': new_value, 'unsafe': p_B_sum, 'feature': new_mu}
				print "1st best candidate"
				print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(theta, expert)
				K = k
				k = (K + KK)/2

			else:
				print "Not the best"
				print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(theta, expert)
				k = (K + k)/2

			print "Add new candidate policy expected feature", new_mu
			
			mu = [[k,  1 - k],[mu_Bs, [expert]]]
			#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			new_index, new_theta, w_delta_mu = expert_update_theta_v1(grids, mu, agent, steps, policies, mus, gamma, epsilon)
			grids.w_features(new_theta)
			print "Weight learnt from combined feature ", new_theta
			new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
			policies = policies + [new_policy]

			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", new_mu
			#mus = mus + [new_mu]
			
			print "Model check ", new_theta
			p_B_sum = output_model(grids, starts, new_policy, steps, safety)
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum
		else:
			print "Model checking failed"
			return None

	print "Iteration ended, best safe theta ", flag['weight']
	print "It's unsafe probability is ", flag['unsafe']
	print "Distance to expert feature ", np.linalg.norm(expert - flag['feature'], ord= 2)
	return grids, flag['weight'], flag['policy'], flag['value']

def expert_synthesize3(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, safety = 0.0001):
	print "Human demo feature ", expert
	print "Initial theta ", start_theta
	flag = []
	index = []
	mu_Bs = []
	mus = []
	policies = []	
	grids.w_features(start_theta)
	start_policy, start_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = policies + [start_policy]
	start_mu = optimal_feature(grids, starts, steps, start_policy, epsilon = epsilon, gamma=gamma)
	expert = start_mu
	#new_mu = start_mu
	#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, start_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	#print "Initial unsafe path rate ", p_B_sum
	outpu_model
	if p_B_sum <= safety:
		return start_theta, start_policy, start_value, start_mu
	while p_B_sum > safety:
		new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
		mu_Bs = mu_Bs + [new_mu_B]
		print "Add counterexample features ", new_mu_B	

		print "Keep generating counterexamples until find a safe candidate" 
		new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
		#new_index, new_theta, w_delta_mu = expert_update_theta(grids, new_mu, agent, steps, policies, mu_Bs, gamma, epsilon)
		grids.w_features(new_theta)
		print "Weight learnt from counterexamples ", new_theta

		new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
		new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
		p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		print "Policy unsafe rate ", p_B_sum
     	print "Found 1st safe policy towards safety ", new_theta	
	#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	mus = mus + [new_mu]
	print "Corresponding feature ", new_mu
	K = 0.5
	mu = 0.5 * new_mu + 0.5 * expert
	print "1st combined feature ", mu
	grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, grids.x_max * grids.y_max, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
	policies = policies + [new_policy]
	grids.w_features(theta)
	print "Weight learnt from 1st combined feature ", theta

	mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "Corresponding feature ", mu

	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "Policy unsafe rate ", p_B_sum

	i = 1	
	while True:
		print ">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", theta, "\nunsafe prob:", p_B_sum, "\nfeature ", mu, " = ", K, "*safe + ", 1 - K, "*expert\n" 
		file = open('log', 'a')
		file.write(">>>>>>>>> " + str(i) + "th iteration, candidate theta: "+ str(theta) + "; unsafe prob: " + str(p_B_sum) + "; feature " + str(mu) + " = " + str(K) + "*safe + " + str(1 - K) + "*expert\n") 
		file.close()
		i = i + 1
		if p_B_sum > safety:
			print "Unsafe, learning from counterexample"
			while p_B_sum > safety:
				new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
				mu_Bs = mu_Bs + [new_mu_B]
				print "Add counterexample feature ", new_mu_B	

				print "Keep generating counterexamples until find a safe candidate" 
				new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
				grids.w_features(new_theta)
				print "Weight learnt from counterexamples ", new_theta

				new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
				exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
				print "Policy unsafe rate ", p_B_sum
			print "Found safe weight towards safety ", new_theta
			new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			mus = mus + [new_mu]
			print "Corresponding feature ", new_mu
		        K = 0.5 * K + 0.5	
			mu = 0.5 * new_mu + 0.5 * mu
			print "combined feature towards safety", mu
			grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			grids.w_features(theta)
			print "Weight learnt from combined feature ", theta
			policies = policies + [new_policy]

			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", mu

			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum
		else:	
			if len(flag) > 0 and np.linalg.norm(flag[-1] - mu) < epsilon:
				print "Difference with the best one is too small", np.linalg.norm(flag[-1] - mu, ord=2)
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
				flag = [theta, new_policy, new_value, mu]
				return grids, flag[0], flag[1], flag[2]
			elif len(flag) > 0 and np.linalg.norm(expert - mu, ord=2)  <= np.linalg.norm(expert - flag[-1], ord=2):
			#elif len(flag) > 0 and np.dot(theta, expert) >= np.dot(flag[0], expert):
				flag = [theta, new_policy, new_value, mu]
				print "New best candidate"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)
			elif len(flag) == 0:
				flag = [theta, new_policy, new_value, mu]
				print "1st best candidate"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)

			else:
				print "Not the best"
				print "Feature deviation from expert ", np.linalg.norm(mu - expert, ord = 2)
				print "Expert can get value ", np.dot(theta, expert)

			print "Add new candidate policy expected feature", mu
			mus = mus + [mu]
			K = 1 - (0.5 + (1 - K) * 0.5) 	
			mu = 0.5 * mu + 0.5 * expert
			print "combined feature towards expert", mu
			grids, theta, new_policy, new_value = expert_train(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			grids.w_features(theta)
			print "Weight learnt from combined feature ", theta
			policies = policies + [new_policy]

			mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", mu
			exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum

	print "Iteration ended, best safe theta ", index[0]
	return grids, flag[0], flag[1], flag[2]

def counterexample(grids, gamma = 0.99, safety = 0.01, epsilon = 1e-5):
	print "Removing last counterexample file"
	os.system('rm counter_example.path')
	while safety > epsilon :
		file = open('grid_world.conf', 'w')
		file.write('TASK counterexample\n')
		file.write('PROBABILITY_BOUND ' + str(safety) + '\n')
		file.write('DTMC_FILE grid_world.dtmc' + '\n')
		file.write('REPRESENTATION pathset' + '\n')
		file.write('SEARCH_ALGORITHM global' + '\n')
		file.write('ABSTRACTION concrete' + '\n')
		file.close()
		cex_comics_timer(['sh', '/home/zekunzhou/workspace/Safety-AI-MDP/comics-1.0/comics.sh', './grid_world.conf'], 5.0)
		try:
			file = open('counter_example.path', 'r')
			print "Generated counterexample for ", safety
			break
		except:
			print "No counterexample found for spec = ", safety, "shrinking down the safey"
			safety = safety / 10.0
	if safety <= epsilon:
		return None
	mu_cex = np.zeros(len(grids.features[-1][-1]))
	total_p = 0
	paths = []
	path_strings = []
	lines = file.readlines()
	file.close()
	for line in range(len(lines)-1):
		path_strings.append(lines[line].split(' ')[0].split('->'))
	for path_string in path_strings:
		path = []
		for state_string in path_string:
			path.append(int(state_string))
		path[0] = float(lines[line].split(' ')[2].split(')')[0])
		paths.append(path)
	for path in range(len(paths)):
		p = paths[path][0]
		mu = np.zeros(len(grids.features[-1][-1]))
		for state in range(1, len(paths[path])):
			y = paths[path][state]/grids.y_max
			x = paths[path][state]%grids.y_max
			mu = mu + (gamma**(state-1)) * grids.features[y][x] 
		mu_cex = mu_cex + p * mu
		total_p = total_p + p
	print "Counterexample for spec = ", safety, ": ",total_p
	print "Counterexample feature ", mu_cex
	mu_cex = mu_cex/total_p
	#os.system('rm counter_example.path')
	return mu_cex		

def output_model(grids, start, policy, steps = None, safety = 0.01):
	if steps is None:
		steps  = grids.y_max * grids.x_max
	print policy
	transitions = []
	target = grids.loc_min_0[0] * grids.x_max + grids.loc_min_0[1]
	for i in range(grids.y_max * grids.x_max):
		y = i / grids.x_max
		x = i % grids.x_max
		k = int(policy[y][x])
		for j in range(grids.y_max * grids.x_max): 	
			yy = j / grids.x_max
			xx = j % grids.x_max
			p = grids.transitions[y, x, k, yy, xx]
			if p > 0.0:
				transitions.append(str(i) + ' ' + str(j) + ' ' + str(p) + '\n') 		
	file = open('grid_world.conf', 'w')
	file.write('TASK counterexample\n')
	file.write('PROBABILITY_BOUND ' + str(safety) + '\n')
	file.write('DTMC_FILE grid_world.dtmc' + '\n')
	file.write('REPRESENTATION pathset' + '\n')
	file.write('SEARCH_ALGORITHM global' + '\n')
	file.write('ABSTRACTION concrete' + '\n')
	file.close()
	
	initial = start[0][0] * grids.x_max + start[0][1]
	file = open('grid_world.dtmc', 'w')
	file.write('STATES ' + str(grids.y_max * grids.x_max) + '\n')
	file.write('TRANSITIONS ' + str(len(transitions)) + '\n')
	file.write('INITIAL ' + str(initial) + '\n')
	file.write('TARGET ' + str(target) + '\n')
	for transition in transitions:
		file.write(transition)
	file.close()
	#prob = model_check_comics_timer()
	

	file = open('optimal_policy', 'w')
	for i in policy:
		for j in i:
			file.write(str(j)+":")
		file.write("\n")
	file.close()
	os.system('/home/zekunzhou/workspace/Safety-AI-MDP/prism-4.4.beta-src/src/demos/run')
	file = open('grid_world.pctl', 'w')
	file.write('P=?[true U<=' + str(steps) + ' x=' + str(grids.loc_min_0[1]) + '&y=' + str(grids.loc_min_0[0])+ ']')
	file.close()
	prob = model_check_prism(['/home/zekunzhou/workspace/Safety-AI-MDP/prism-4.4.beta-src/bin/prism', './grid_world.pm', './grid_world.pctl'], 5.0)
	
	if prob is not None:
		return prob
	else:
		return None

def model_check_prism(cmd = ['/home/zekunzhou/workspace/Safety-AI-MDP/prism-4.4.beta-src/bin/prism', './grid_world.pm', './grid_world.pctl'], timeout_sec = 5.0):
	kill_proc = lambda p: p.kill()
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
  	try:
    		timer.start()
   	 	stdout, stderr = proc.communicate()
  	finally:
    		timer.cancel()
		#print stdout
		print stderr
  		try:
			lines = "".join(stdout).split('\n')
			for line in lines:
				if line.split(':')[0] == 'Result':
					prob = float(line.split(':')[1].split('(')[0])
					break
			#prob = float("".join(stdout).split('\n')[-7].split(' ')[1])
			if prob <= 1.0:
  				return prob
		except:
			print "PRISM model checking failed, return None"
			return None
	
	
		
def model_check_comics_timer(cmd = ['sh', '/home/zekunzhou/workspace/Safety-AI-MDP/comics-1.0/comics.sh', './grid_world.conf', '--only_model_checking'], timeout_sec = 5.0):
  	kill_proc = lambda p: os.system('kill -9 $(pidof comics)')
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
  	try:
    		timer.start()
   	 	stdout, stderr = proc.communicate()
  	finally:
    		timer.cancel()
		#print stdout
		print stderr	
  		try:
			prob = float("".join(stdout).split('\n')[-2].split(':')[-1])
			if prob <= 1.0:
  				return prob
		except:
    			prob = float("".join(stdout).split('\n')[-3].split(':')[-2].split(";")[0])
			if prob <= 1.0:
				return prob
		return None

  	
def cex_comics_timer(cmd = ['sh', '/home/zekunzhou/workspace/comics-1.0/comics.sh', './grid_world.conf'], timeout_sec = 5.0):
  	kill_proc = lambda p: os.system('kill -9 $(pidof comics)')
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
  	try:
    		timer.start()
   	 	stdout, stderr = proc.communicate()
  	finally:
    		timer.cancel()
		#print stdout
		print stderr	


	
class train:
	def __init__(self, grids= grid(12, 12, 0.6), starts = None, steps = float("inf"), epsilon = 1e-4, gamma = 0.6, iteration = 30, theta = np.array([1./3., 1./3., -3./3., 0.0])):
		if steps is None:
			self.steps= float("inf")
		else:
			self.steps = steps
		self.iteration = iteration
		self.epsilon = epsilon
		self.gamma = gamma	
		self.grids=grids
		self.expert= []
		self.expert_policy = np.zeros([self.grids.y_max, self.grids.x_max])
		self.demo_policy = np.zeros([self.grids.y_max, self.grids.x_max])
		self.agent=car(states=self.grids.states)
		self.theta = np.array(theta)
		self.starts = starts
		#pylab.ioff()
		
			
	def synthesize(self, theta, starts= None, epsilon= None, safety = 0.01):
		if epsilon is None:
			epsilon = self.epsilon
		policies = []
		mus = []
		exp_mu = np.zeros(len(self.grids.features[-1][-1]))	
		if theta is None:
			theta = self.theta
		theta = np.array(theta).astype(float)/np.linalg.norm(theta, ord=2)
		'''
		self.expert_policy, exp_mu = real_optimal(self.grids, self.agent, starts = starts, steps = self.steps, theta = theta, gamma = self.gamma, epsilon = epsilon)
		print "real optimal policy"
		print self.expert_policy
		print "real expected feature under optimal policy"
		print exp_mu
		demo_mu = exp_mu
		exp_u_G, p_G, exp_u_B, p_B,_ = sample_feature(self.grids, self.agent, starts, self.grids.x_max * self.grids.y_max, self.expert_policy, epochs= 10000, epsilon = epsilon, gamma=self.gamma)
		safety = np.sum(np.reshape(p_B, [self.grids.y_max * self.grids.x_max, 1]), 0)
		print "Optimal policy unsafe rate ", safety
		'''

		#print "unsafe path feature ", np.sum(np.reshape(exp_u_B, [self.grids.y_max * self.grids.x_max, len(self.grids.features[-1][-1])]), 0)
		#demo_mu = optimal_feature(self.grids, starts, self.steps, self.demo_policy, epsilon = self.epsilon, gamma=self.gamma)
		#For [3 ,3], [6, 6], [4, 4], [7, 7]
		#demo_mu = np.array([ 40.89036619,   0.64147873,   9.98327114,   0.])
		#For [3, 7], [7,3], [5, 5], [7, 7]
		#demo_mu = np.array([ 44.075131,    43.41778817,   5.40566983,   1.61532099])
		#demo_mu = np.array([ 44.075131,    43.41778817,   5.40566983,  0.0])
		#For [3, 7], [1, 6], [1, 3], [6, 5]
		#demo_mu = np.array([49.06798858,  50.84211635,   3.93830203,   0.        ])
		#theta = np.array([0.69683357, 0.71605115, 0.04115484, 0.0])
		#For [3, 7], [1, 6], [1, 3], [4, 4]
		#demo_mu = np.array([ 22.40729888,  62.27166007,   5.2054007,    2.57517056])
		#theta = np.array([ 0.70259415,  0.69168086, -0.06375435,  0.15451353])

		#For [1, 7], [7, 2], [3, 3], [4, 4]
		#demo_mu = np.array([44.79990088, 45.11285122,   1.60969386,   0.        ])	
		#safety = 0.005
		#theta = np.array([ 0.71021283,  0.70381664, -0.01548792,  0.])
		#For [2, 6], [1, 6], [3, 3], [4, 4]
		#demo_mu = np.array([ 63.22198067,  63.41995431,   3.7295239,    0.        ])
		#For [7, 7], [7, 6], [4, 4], [4, 6]
		#demo_mu = np.array([ 53.45346621,  51.51237465,   3.10056466,   5.0122147 ])
		#For [7, 7], [7, 6], [4, 6], [5, 3]
		#demo_mu = np.array([ 58.95770271,  59.17225335,   3.50161782,   3.86107397])
		#For [7, 7], [7, 6], [4, 6], [6, 1]
		#demo_mu = np.array([ 57.8561493,   57.68399681,   3.19279245,   4.25976904])
		#theta = np.array([ 0.72613264,  0.64114756, -0.05041496,  0.24314507])

		#self.demo_policy, _  = real_optimal(self.grids, self.agent, starts = starts, steps = self.steps, theta = theta, gamma = self.gamma, epsilon = epsilon)
		safety = 0.01
		print "safety requirement is ", safety

		if starts is None:
			starts = np.array(self.starts)
		if theta is None:
			theta = self.theta/np.linalg.norm(self.theta, ord=2)
		if epsilon is None:
			epsilon = self.epsilon
		again = 'y'
		while(again != 'n' and again!= 'N'):
			if again != 'y' and again!= 'Y':
				print "Invalid input, exit...??"
				break
			else:
				start=starts[random.randint(0, len(starts)-1)]
				expert_temp, again = demo(self.grids, self.agent, start, steps= self.steps, gamma=self.gamma, epsilon= epsilon)
				self.expert.append(expert_temp)
		starts = []
		print "Start training..."
			
		demo_mu = np.zeros(4)
		for i in range(len(self.expert)):
			demo_mu = demo_mu + self.expert[i]["mu"] 
			starts.append(self.expert[i]["trajectory"][0]["state"])
		demo_mu = demo_mu/len(self.expert)	

		print "expected demo mu is ", demo_mu
		_, theta, self.demo_policy,_ = expert_train(self.grids, demo_mu, self.agent, epsilon = epsilon, starts = starts, steps= self.steps, iteration = 10, gamma=self.gamma, start_theta = None, MC = False)
		print "theta learnt from demo mu ", theta
		file = open('log', 'a')	
		file.write("\nhuman demo for ultimate synthesis\n")
		file.write(str(self.grids.loc_max_0))
                file.write(str(self.grids.loc_max_1))
                file.write(str(self.grids.loc_min_0))
                file.write(str(self.grids.loc_min_1)+'\n')

		file.write("parameter "+ str(theta) + "\n")
		for i in self.demo_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.write("\n")

		prob = output_model(self.grids, starts, self.demo_policy, self.steps, safety)
		print "model output finished ", prob
		mu_cex =  counterexample(self.grids, self.gamma, safety, epsilon)
		if mu_cex is not None:
			print mu_cex
		_, theta, self.demo_policy,_ = expert_synthesize2(self.grids, demo_mu, self.agent, epsilon = epsilon, starts = starts, steps= self.steps, iteration = self.iteration, gamma=self.gamma, start_theta = theta, MC = False, safety = safety)
		#draw_grids(self.grids.rewards, None)
		file = open('log', 'a')	
		file.write("\nleanrt from ultimate synthesis\n")
		file.write(str(self.grids.loc_max_0))
                file.write(str(self.grids.loc_max_1))
                file.write(str(self.grids.loc_min_0))
                file.write(str(self.grids.loc_min_1)+'\n')

		file.write("parameter "+ str(theta) + "\n")
		for i in self.demo_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.write("feature " + str(demo_mu) + "\n")
		print self.demo_policy



 	def learn_from_policy(self, starts = None, expert_policy = None, safety = True):
		if starts is None:
			starts = np.array(self.starts)
		if expert_policy is not None:
			self.expert_policy = expert_policy
		else:
			i = 0
                        j = 0
                        file = open('demo_policy', 'r')
                        for line in file:
                                for j in range(len(line.split(":"))-1):
                                       	self.expert_policy[i, j] = float(line.split(":")[j])
                                i = i + 1
                        file.close()
		
		print self.expert_policy	

		
		file=open('log', 'a')
		file.write("learn from human policy\n")
		
		file.write(str(self.grids.loc_max_0))
                file.write(str(self.grids.loc_max_1))
                file.write(str(self.grids.loc_min_0))
                file.write(str(self.grids.loc_min_1)+'\n')

		for i in range(len(self.expert_policy)):
			for j in range(len(self.expert_policy[i])):
				file.write(str(self.expert_policy[i, j]) + ":")
			file.write("\n")
		exp_mu = optimal_feature(self.grids, starts, self.steps, self.expert_policy, epsilon = self.epsilon, gamma=self.gamma)
		print "analytical feature" + str(exp_mu)
		if safety is True:
                	exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(self.grids, self.agent, starts, self.steps, self.expert_policy, epochs= 10000, epsilon = self.epsilon, gamma=self.gamma)
			print exp_u_G[0][0]
                        p_B_expert = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
			mu_B = np.sum(np.reshape(exp_u_B, [self.grids.y_max*self.grids.x_max, len(self.grids.features[-1][-1])])* np.reshape(p_B, [self.grids.y_max*self.grids.x_max, 1]), 0)
			mu_G = np.sum(np.reshape(exp_u_G, [self.grids.y_max*self.grids.x_max, len(self.grids.features[-1][-1])]), 0)
			print "expected failure path feature ", mu_B
			print "expected successful path feature ", mu_G
			exp_mu = mu_G
		#exp_u_G, p_G, exp_u_B, p_B = sample_feature(self.grids, self.agent, starts, self.grids.x_max * self.grids.y_max, self.expert_policy, epochs= 5000, epsilon = 1e-3, gamma=self.gamma)
		#exp_mu = np.sum(np.reshape(exp_u_G, [self.grids.y_max * self.grids.x_max, len(self.grids.features[-1][-1])]), 0)
		#print "monte carlo " + str(exp_mu)
		file.write(str(exp_mu) + '\n')
		file.close()	
	
		_, theta, policy, _= expert_train(self.grids, exp_mu, self.agent, starts = starts, steps= self.steps, epsilon= self.epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= None, MC = False, safety = None)
		for i in range(len(policy)):
			for j in range(len(policy[i])):
				if policy[i, j] != self.expert_policy[i, j]:
					print "feature matched policy is different with expert"
					'''
					exp_u_G, p_G, exp_u_B, p_B = sample_feature(self.grids, self.agent, starts, self.steps, policy, epochs= 5000, epsilon = self.epsilon, gamma=self.gamma)
					p_B_sum = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
					print "feature matched policy unsafe rate ", p_B_sum						   '''
					file = open('log', 'a')
					file.write("learnt policy is different\n")
					#file.write(str(p_B_expert)+'\n')	
					file.write(str(theta)+'\n')
					for i in policy:
						for j in i:
							file.write(str(j)+":")
						file.write("\n")
					#file.write(str(p_B_sum)+'\n')
				
					file.close()
					return policy
		print "precisely learnt"
		file = open('log', 'a')
		file.write("precisely learnt\n")
		file.close()
		return policy

		

	def real_expert_train(self, starts = None, expert_theta = None, epsilon= None, distribution= None, safety = True):
		if distribution is None:
			distribution = [1.0]

		if starts is None:
			if self.starts is None:
				starts = [np.array([0, 0])]
			else:
				starts = self.starts
		if epsilon is None:
			epsilon = self.epsilon
		if expert_theta is None:
			expert_theta = self.theta/np.linalg.norm(self.theta, ord=2)
		else:
			expert_theta = theta/np.linalg.norm(theta, ord=2)
		print "feature states are "+ str(self.grids.loc_max_0)+str(self.grids.loc_max_1)+str(self.grids.loc_min_0)+str(self.grids.loc_min_1)
		print "ground true weight is ", expert_theta
		self.expert_policy, exp_mu =real_optimal(self.grids, self.agent, starts = starts,  steps = self.steps, theta = expert_theta, gamma=self.gamma, epsilon = self.epsilon)		
		#print self.grids.rewards
		print "expert expected feature counts:"
		print exp_mu
	
		if safety is True:
			safety = 0.01
		prob = output_model(self.grids, starts, self.expert_policy, safety)
		print "model output finished ", prob
		mu_cex =  counterexample(self.grids, self.gamma, safety, epsilon)
		if mu_cex is not None:
			print mu_cex
		else:
			print "No counterexample found. Comics says that it's safe!!!!!!!!!!!"
			

		file = open('log', 'a')
		file.write(str(self.grids.loc_max_0))
		file.write(str(self.grids.loc_max_1))
		file.write(str(self.grids.loc_min_0))
		file.write(str(self.grids.loc_min_1)+'\n')
		file.write(str(expert_theta) + '\n')
		file.write(str(exp_mu) + '\n')
		file.close()

		for prob in distribution:
			print prob, " optimal expert is teaching"
			for i in range(len(self.expert_policy)):
				for j in range(len(self.expert_policy[i])):
					if random.random() >= prob:
						actions = [0.0, 1.0, 2.0, 3.0, 4.0]
						random.shuffle(actions)
						if actions[0] == self.expert_policy[i, j]:
							self.demo_policy[i, j]=actions[1]
						else:
							self.demo_policy[i, j]=actions[0]
					else:
						self.demo_policy = self.expert_policy
		       	 				
			demo_mu = optimal_feature(self.grids, starts, self.steps, self.expert_policy, epsilon = self.epsilon, gamma=self.gamma)
			if safety is True:
                		exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(self.grids, self.agent, starts, self.steps, self.expert_policy, epochs= 10000, epsilon = self.epsilon, gamma=self.gamma)
                        	p_B_expert = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
				mu_B = np.sum(np.reshape(exp_u_B, [self.grids.y_max*self.grids.x_max, len(self.grids.features[-1][-1])])* np.reshape(p_B, [self.grids.y_max*self.grids.x_max, 1]), 0)
				mu_G = np.sum(np.reshape(exp_u_G, [self.grids.y_max*self.grids.x_max, len(self.grids.features[-1][-1])]), 0)
				print "expected failure path feature ", mu_B
				#demo_mu = (exp_mu - mu_B)/(1 - p_B_expert)
				demo_mu = mu_G
				print "expected succesful path feature ", demo_mu
                                file = open('log', 'a')
                                file.write("policy future reach unsafe state rate "+ str(p_B_expert) + "\n") 
                                print "policy future reach unsafe state rate ", p_B_expert
                                file.close()      

				
			file = open('optimal_policy', 'w')
			for i in self.demo_policy:
				for j in i:
					file.write(str(j)+":")
				file.write("\n")
			file.close()


			file = open('log', 'a')
			file.write(str(prob) + " optimal expert is teaching\n")
			file.write(str(demo_mu) + " is the expected feature\n") 
			for i in self.demo_policy:
				for j in i:
					file.write(str(j)+":")
				file.write("\n")
			file.close()
			_, theta, policy, _= expert_train(self.grids, demo_mu, self.agent, starts = starts, steps= self.steps, epsilon= epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= -1.0* expert_theta, MC = False, safety= None)

			unmatch = False
			for i in range(len(policy)):
				for j in range(len(policy[i])):
					if policy[i, j] != self.demo_policy[i, j]:
						print "feature matched policy is different with ", prob, " expert"
						file = open('log', 'a')

	  			  	        file.write("feature matched policy is different with " + str(prob) +" expert\n")
						
						file.write("learnt parameter " + str(theta)+'\n')
						for m in policy:
							for n in m:
								file.write(str(n)+":")
							file.write("\n")
						file.close()
						unmatch = True
						break
				if unmatch is True:
					break
				
			if unmatch is False:
				print "precisely learnt"
				file = open('log', 'a')
				file.write("learnt parameter " + str(theta)+'\n')
				file.write("policy precisely learnt\n")
				file.close()

			
		'''
		if safety is not None:
			safety = p_B_expert
		print "original expert policy's unsafe rate ", p_B_expert

		file = open('expert_policy', 'w')
		for i in self.expert_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.close()
		
		policy_temp = np.array(self.expert_policy)
		policy_temp[self.grids.loc_min_0[0]-1, self.grids.loc_min_0[1]-1] = 3
		policy_temp[self.grids.loc_min_0[0]-1, self.grids.loc_min_0[1]]=1
		policy_temp[self.grids.loc_min_0[0]-1, self.grids.loc_min_0[1]+1] = 1
		policy_temp[self.grids.loc_min_0[0], self.grids.loc_min_0[1]-1] = 2
		policy_temp[self.grids.loc_min_0[0], self.grids.loc_min_0[1]+1] = 1
		policy_temp[self.grids.loc_min_0[0]+1, self.grids.loc_min_0[1]-1] = 2
		print policy_temp
		mu_temp = optimal_feature(self.grids, starts, self.steps, policy_temp, epsilon = self.epsilon, gamma=self.gamma)
		print "hand-modified policy feature error", np.linalg.norm(exp_mu-mu_temp, ord=2)
		exp_u_G, p_G, exp_u_B, p_B = sample_feature(self.grids, self.agent, starts, self.steps, policy_temp, epochs= 5000, epsilon = self.epsilon, gamma=self.gamma)
		p_B_sum = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
		print "new policy unsafe rate ", p_B_sum
			

		pylab.title('Real Reward. Try real expert? Type [y/n] in the terminal')
		draw_grids(self.grids.rewards, None)

		pylab.ion()
		pylab.title('Rewards from expert train, close to continue')
		'''
		return self.expert_policy


		_, theta, policy, _= expert_train(self.grids, exp_mu, self.agent, starts = starts, steps= self.steps, epsilon= epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= -1.0* expert_theta, MC = False, safety= None)
		for i in range(len(policy)):
			for j in range(len(policy[i])):
				if policy[i, j] != self.expert_policy[i, j]:
					print "feature matched policy is different with expert"
					'''
					exp_u_G, p_G, exp_u_B, p_B = sample_feature(self.grids, self.agent, starts, self.steps, policy, epochs= 5000, epsilon = self.epsilon, gamma=self.gamma)
					p_B_sum = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
					print "feature matched policy unsafe rate ", p_B_sum						   '''
		return policy
	
	
	def human_train(self, starts = None, expert_theta = None, epsilon= None):
		if starts is None:
			starts = np.array(self.starts)
		if expert_theta is None:
			expert_theta = self.theta/np.linalg.norm(self.theta, ord=2)
		if epsilon is None:
			epsilon = self.epsilon
		
		#file = open('demo_policy', 'w')
		#for i in starts:
		#	file.write(str(i[0])+","+str(i[1])+":")	
		#file.write("\n")				 
		#file.close()

		again = 'y'
		while(again != 'n' and again!= 'N'):
			if again != 'y' and again!= 'Y':
				print "Invalid input, exit...??"
				break
			else:
				start=starts[random.randint(0, len(starts)-1)]
				expert_temp, again = demo(self.grids, self.agent, start, steps= self.steps, gamma=self.gamma, epsilon= epsilon)
				self.expert.append(expert_temp)
		starts = []
		print "Start training..."
			
		demo_mu = np.zeros(4)
		for i in range(len(self.expert)):
			demo_mu = demo_mu + self.expert[i]["mu"] 
			starts.append(self.expert[i]["trajectory"][0]["state"])
		demo_mu = demo_mu/len(self.expert)	
		print "expected demo mu is ", demo_mu

		_, theta, self.demo_policy,_ = expert_train(self.grids, demo_mu, self.agent, epsilon = epsilon, starts = starts, steps= self.steps, iteration = self.iteration, gamma=self.gamma, start_theta = None, MC = False)
		draw_grids(self.grids.rewards, None)
		file = open('log', 'a')	
		file.write("leanrt from human demo\n")
		file.write(str(self.grids.loc_max_0))
                file.write(str(self.grids.loc_max_1))
                file.write(str(self.grids.loc_min_0))
                file.write(str(self.grids.loc_min_1)+'\n')

		file.write("parameter "+ str(theta) + "\n")
		for i in self.demo_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.close()
		
		while True:
			real=raw_input("Try modified policy? [Y/N]")
			if real == 'Y' or real == 'y':
				i = -2
				j = 0
				file = open('demo_policy', 'r')
				for line in file:
					if i == 0: 
						for j in range(len(line.split(":"))-1):
							self.demo_policy[i, j] = float(line.split(":")[j])
					i = i + 1			
				file.close()
				mu_temp = optimal_feature(self.grids, starts, self.steps, self.demo_policy, epsilon = self.epsilon, gamma=self.gamma)
				print "modified policy has feature error", np.linalg.norm(demo_mu-mu_temp, ord=2)

				_, theta, policy, _= expert_train(self.grids, mu_temp, self.agent, starts = starts, steps= self.steps, epsilon= epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= None, MC = False, safety = None)
				for i in range(len(policy)):
					for j in range(len(policy[i])):
						if policy[i, j] != self.demo_policy[i, j]:
							print "And not so well learnt the modified policy"
							i = len(policy)
							break
				print "parameter learnt from modified policy is ", theta
			else:
				break

		return self.demo_policy



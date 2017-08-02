from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab
import warnings
import random
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
	while(steps > 0):
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
		except:
			print("Invalid action, input again")
			next
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(grids.transitions, action)})
		trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0]][trajectory[-1]["state"][1]])
		grids.w_features(theta)
		draw_grids(grids.rewards, trajectory)

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

	


def sample_feature(grids, agent, starts, steps, policy, epochs = 1000, epsilon = 1e-5, gamma = 0.5):
	u =  np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	u_G = np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	u_B = np.zeros([len(grids.features), len(grids.features[-1]), len(grids.features[-1][-1])])
	p_B = np.zeros([len(grids.features), len(grids.features[-1])])
	p_G = np.zeros([len(grids.features), len(grids.features[-1])])

	print starts
	for start in range(len(starts)):
		print "sampling ", starts[start]
	 	epochs_G=0.0
	 	epochs_B=0.0
 	 	while epochs_G + epochs_B <= epochs-1:
			agent.state = np.array(starts[start])
			u_epoch = grids.features[agent.state[0]][agent.state[1]] 
			steps = 0
			end = False
			fail = False
			while end is False:
				steps = steps + 1
				action_epoch = int(policy[agent.state[0], agent.state[1]])
				agent.state = np.array(agent.move(grids.transitions, action_epoch))
				if agent.state[0] ==  grids.loc_min_0[0] and agent.state[1] == grids.loc_min_0[1]:
					fail = True
				u_epoch = u_epoch + grids.features[agent.state[0]][agent.state[1]] * (gamma**steps)
				if np.linalg.norm(grids.features[agent.state[0]][agent.state[1]] * (gamma**steps), ord=2) <= epsilon:
					end = True
			if fail is True:
				u_B[starts[start][0], starts[start][1]] = u_B[starts[start][0], starts[start][1]]  + u_epoch
				epochs_B=epochs_B+1
			else:							
				u_G[starts[start][0], starts[start][1]]  = u_G[starts[start][0], starts[start][1]]  + u_epoch
				epochs_G=epochs_G+1
			if np.linalg.norm(u_epoch/(epochs_G+epochs_B), ord=2) <= epsilon and  epochs_G+epochs_B>=epochs/10:
				break
	
		u_B[starts[start][0], starts[start][1]]  = u_B[starts[start][0], starts[start][1]] /epochs_B
		u_G[starts[start][0], starts[start][1]]  = u_G[starts[start][0], starts[start][1]] /epochs_G
		if epochs_B <= 0.0:
			u_B[starts[start][0], starts[start][1]] =np.zeros(len(grids.features[-1][-1]))
		if epochs_G <= 0.0:
			u_G[starts[start][0], starts[start][1]] =np.zeros(len(grids.features[-1][-1]))
		p_B[starts[start][0], starts[start][1]]  = float(epochs_B/(epochs_B+epochs_G))
  		p_G[starts[start][0], starts[start][1]]  = float(epochs_G/(epochs_B+epochs_G))
	return u_G, p_G, u_B, p_B





def optimal_feature(grids, starts, steps, policy, epsilon = 1e-5, gamma= 0.5):
	exp_u= np.zeros(len(grids.features[-1][-1]))
	features= np.array(grids.features)
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

	diff = float("inf")
	while diff > epsilon and steps + 1 == steps:
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
		exp_u_G, p_G, exp_u_B, p_B = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
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
			exp_u_G, p_G, exp_u_B, p_B = sample_feature(grids, agent, starts, steps, new_policy, epochs= 1000, epsilon = 1e-3, gamma=gamma)
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
				exp_u_G, p_G, exp_u_B, p_B = sample_feature(grids, agent, starts, steps, policies[index], epochs= 5000, epsilon = epsilon, gamma=gamma)
				p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
				print "best policy's unsafe rate ", p_B_sum
				if p_B_sum > safety:
					mus[index] = mus[index] -  (p_B_sum - safety) * np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0)
				print "feature counts modified to ", mus[index]

				
		if abs(w_delta_mu) < epsilon:
			print "|expert_w_mu - w_mu| = ", abs(w_delta_mu), " < ", epsilon
			#index = new_index
			break	

	
	#index = -1
	#print "best policy"
	#print policies[index]
	print "best weight", thetas[index]
	print "best feature", mus[index]
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
		G_i=[[], [], [], []]
		h_i = []
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
		dims = {'l': len(mus), 'q': [5], 's': []}
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

	
def expert_update_theta_1(grids, expert, agent, steps, policies, gamma=0.5, epsilon = 1e-5):

	mus=[]
	delta_mus = []
	w_delta_mus=[]
	solutions=[]
	exp_mu = expert
	for policy in policies:
		mu = optimal_feature(grids, steps, policy, epsilon = epsilon, gamma=gamma)
		mus.append(mu)
	A= matrix(np.transpose(np.array(mus)))
	b= matrix(np.transpose(np.array(exp_mu)))
	m, n = A.size
	I = matrix(0.0, (n+1,n))
	I[::n+2] = -1.0
	I[n::n+1] = 1.0
	G = matrix(I)
	h = matrix(n*[0.0] + [1.0])
	dims = {'l': n+1, 'q': [], 's': []}
	sol = solvers.coneqp(A.T*A, -A.T*b, G, h, dims)	
	sol['status']
	solution = np.array(sol['x'])
	solution = np.reshape(solution, [len(solution)])
	delta_mu=np.array(exp_mu) - np.dot(np.transpose(np.array(mus)), np.transpose(solution))
	weight = delta_mu/np.linalg.norm(delta_mu, ord=2)

	#solution = delta_mus[index]/np.linalg.norm(delta_mus[index], ord =2)
	#delta_mu = np.linalg.norm(delta_mus[index], ord =2)  
	return weight

def multi_learn(grid, agent, theta, exp_policy, exp_mu, starts=None, steps=float("inf"), epsilon=1e-4, iteration=20, gamma=0.9, safety = 0.02):
	end = False
	print "starting multiple goal learning"
	value = np.array(grid.rewards)
	exp_u_G, p_G, exp_u_B, p_B = sample_feature(grid, agent, starts, steps, exp_policy, epochs= 5000, epsilon = epsilon, gamma=gamma)
	p_B_sum = np.sum(np.reshape(p_B, [grid.y_max*grid.x_max]))/(len(starts))
	print "when feature matched, unsafe rate ", p_B_sum
	while end is False:
		print value
		for i in range(len(p_B)):
			for j in range(len(p_B[i])):
				value[i, j] = value[i, j] * (1.0 - p_B[i, j])
		print value
		new_policy = update_policy(grid, value, epsilon= epsilon, gamma=gamma)
		print new_policy
		exp_u_G, p_G, exp_u_B, p_B = sample_feature(grid, agent, starts, steps, new_policy, epochs= 5000, epsilon = epsilon, gamma=gamma)			
		p_B_sum = np.sum(np.reshape(p_B, [grid.y_max*grid.x_max]))/(len(starts))
		print "afet 1 time safety upgrade, unsafe rate ", p_B_sum
		new_mu = optimal_feature(grid, starts = starts, steps=steps, policy=new_policy, epsilon = epsilon, gamma=gamma)
		print "feature error ", np.linalg.norm(new_mu-exp_mu, ord=2)
		_, new_theta, new_policy, _= expert_train(grid, new_mu, agent, starts = starts, steps=steps, epsilon=epsilon, iteration=iteration, gamma=gamma, start_theta= None, MC = False, safety = None)
		print "after 1 time safety upgrade, learnt policy"
		print new_policy
	
			
		if np.linalg.norm(new_mu-exp_mu, ord=2) < epsilon and p_B_sum <= safety:
			end = True
		else:
			theta = new_mu/np.linalg.norm(new_mu, ord=2) 
			grid, theta, policy, value= expert_train(grid, exp_mu, agent, starts = starts, steps= steps, epsilon= epsilon, iteration=iteration, gamma=gamma, start_theta= theta, MC=True)
			exp_u_G, p_G, exp_u_B, p_B = sample_feature(grid, agent, starts, steps, policy, epochs= 5000, epsilon = epsilon, gamma=gamma)
			print p_B
			print "unsafe rate ", np.sum(np.reshape(p_B, [grid.y_max*grid.x_max]))/(len(starts))
			print "when feature matched, unsafe rate ", p_B_sum
			if p_B_sum < safety:
				end = True
			
		#multi_learn(grid, agent, thetas, policies, mus, votes, starts, steps, epsilon, iteration, gamma, safety)
	
	return policy



	
class train:
	def __init__(self, grids= grid(12, 12, 0.6), starts = None, steps = float("inf"), epsilon = 1e-4, gamma = 0.6, iteration = 30, theta = np.array([1./3., 1./3., -3./3., 0.0])):
		if steps is None:
			self.steps=float("inf")
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
		
			
	def synthesize(self, theta, starts= None, epsilon= None):
		if epsilon is None:
			epsilon = self.epsilon
		policies = []
		mus = []
		exp_mu = np.zeros(len(self.grids.features[-1][-1]))	

		theta = np.array(theta).astype(float)/np.linalg.norm(theta, ord=2)
		self.expert_policy, exp_mu = real_optimal(self.grids, self.agent, starts = starts, steps = self.steps, theta = theta, gamma = self.gamma, epsilon = epsilon)
		print "real optimal policy"
		print self.expert_policy
		print "real expected feature under optimal policy"
		print exp_mu
		'''
		file = open('synthesize_policy', 'w')
		for i in policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.close()
		
		real=raw_input("Model checking finished? [Y/N]")
			if real == 'Y' or real == 'y':
				file = open('synthesize_policy', 'w')
				for i in file:
					
				file.close()
		'''
		policy = multi_learn(self.grids, self.agent, theta, self.expert_policy, exp_mu, starts = starts, steps= self.steps, epsilon= self.epsilon, iteration=self.iteration, gamma=self.gamma, safety = 0.01)



	def learn_from_policy(self, starts = None, expert_policy = None):
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
		file.close()	
	
		exp_mu = optimal_feature(self.grids, starts, self.steps, self.expert_policy, epsilon = self.epsilon, gamma=self.gamma)
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

		

	def real_expert_train(self, starts = None, expert_theta = None, epsilon= None, distribution= None, safety = False):
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

		file = open('log', 'a')
		file.write(str(self.grids.loc_max_0))
		file.write(str(self.grids.loc_max_1))
		file.write(str(self.grids.loc_min_0))
		file.write(str(self.grids.loc_min_1)+'\n')
		file.write(str(expert_theta) + '\n')
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

			if safety is True:                                                            
                                exp_u_G, p_G, exp_u_B, p_B = sample_feature(self.grids, self.agent, starts, self.steps, self.demo_policy, epochs= 10000, epsilon = self.epsilon, gamma=self.gamma)                                                                                 
                                p_B_expert = np.sum(np.reshape(p_B, [self.grids.y_max*self.grids.x_max]))/(len(starts))
                                file = open('log', 'a')
                                file.write("policy future reach unsafe state rate "+ str(p_B_expert) + "\n") 
                                file.close()      

			file = open('log', 'a')
			file.write(str(prob) + " optimal expert is teaching\n")
			for i in self.demo_policy:
				for j in i:
					file.write(str(j)+":")
				file.write("\n")
			file.close()
			
			file = open('optimal_policy', 'w')
			for i in self.demo_policy:
				for j in i:
					file.write(str(j)+":")
				file.write("\n")
			file.close()

			demo_mu = optimal_feature(self.grids, starts, self.steps, self.demo_policy, epsilon = self.epsilon, gamma=self.gamma)
			_, theta, policy, _= expert_train(self.grids, demo_mu, self.agent, starts = starts, steps= self.steps, epsilon= epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= -1.0* expert_theta, MC = False, safety= None)

			unmatch = False
			for i in range(len(policy)):
				for j in range(len(policy[i])):
					if policy[i, j] != self.demo_policy[i, j]:
						print "feature matched policy is different with ", prob, " expert"
						file = open('log', 'a')

	  			  	        file.write("feature matched policy is different with " + str(prob) +" expert\n")
						file.write(str(theta)+'\n')
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



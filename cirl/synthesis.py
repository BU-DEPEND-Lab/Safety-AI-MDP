from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab
import warnings
warnings.filterwarnings("ignore")

def real_optimal(grids, agent, steps, theta = None, gamma=0.5, epsilon = 1e-5):
	expert=[]
	if theta == None:
		theta = np.array([1./3., 1./3., -3./3.])
	theta = theta/np.linalg.norm(theta, ord=2)
	grids.w_features(theta)
	optimal_policy= update_policy(grids, steps= steps, epsilon= epsilon, gamma= gamma)
	print "real optimal policy generated"
	print optimal_policy
	file = open('optimal_policy', 'w')
	for i in optimal_policy:
		for j in i:
			file.write(str(j)+":")
		file.write("\n")
	file.close()

	opt_u = optimal_feature(grids, steps, optimal_policy, epsilon, gamma)
	return opt_u
	


def demo(grids, agent, start, steps, theta = None, gamma=0.5, epsilon = 1e-5):
	expert={}
	agent.state=np.array(grids.states[start[0], start[1]])
	if theta == None:
		theta=np.array([1./3., 1./3., -3./3.])
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0], agent.state[1]]}]
	grids.w_features(theta)
	pylab.close()
	pylab.ion()
	pylab.title("Generate demonstration[0:end, 1: left, 2: down, 3: right, 4: up]")
	draw_grids(grids.rewards, trajectory)
	print grids.rewards
	mu=np.zeros(3)
	while(steps > 0):
		action = input("%0.0f steps left, action is " % steps)
		if steps == float("inf") and action == 0:
			pylab.ioff()
			pylab.close('all')
			break
		steps = steps - 1	
		if action!= 0 and action != 1 and action !=2 and action !=3 and action !=4:
			print("Invalid action, input again")
			next
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(grids.transitions, action)})
		trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0], trajectory[-1]["state"][1]])
		grids.w_features(theta)
		draw_grids(grids.rewards, trajectory)

	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	expert["mu"]=mu
	expert["trajectory"]=trajectory
 	playagain=raw_input("Want to play again? [y/n]?")
	return expert, playagain
	
def calc_u(grids, agent, policy, steps, gamma=0.5):
	mu=np.zeros(3)
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0], agent.state[1]]}]
	for i in range(steps):
		action=policy[agent.state[0], agent.state[1]]
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(grids.transitions, action)})
		trajectory[-1]["feature"]=np.array(grids.features[trajectory[-1]["state"][0], trajectory[-1]["state"][1]])
	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	return mu, trajectory

def exp_u(grids, agent, policy, start, start_action=None, steps=None, epoch=1000, gamma=0.5):
	if steps == None:
		steps = epoch
	mu=np.zeros([3])
	agent.state=np.array([start[0], start[1]])
	trajectory_i_j={}
	for i in range(epoch):
		if start_action != None:
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

	
def update_theta(grids, expert, agent, policies, start, steps=None, epoch=1000, penalty=-0.5, gamma=0.5):
	if steps == None:
		steps = len(expert["trajectory"])
	if start == None:
		start = np.array(expert["trajectory"][0]["state"])
	mus=[]
	exp_mu=expert["mu"]
	for policy in policies:
		mu = exp_u(grids=grids, agent=agent, policy=policy, start=start, steps=steps, epoch=epoch, start_action=None)
		mus.append(mu)
	w_delta_mus=[]
	solutions=[]
	for i in range(len(mus)):
		G_i=[[], [], []]
		h_i = []
		c = matrix(mus[i] - exp_mu)
		for j in range(len(mus)):
			G_i[0].append(-1 * mus[i][0] - (-1) * mus[j][0])
			G_i[1].append(-1 * mus[i][1] - (-1) * mus[j][1])
			G_i[2].append(-1 * mus[i][2] - (-1) * mus[j][2])
			h_i.append(0)
		G_i[0].append(-1. * exp_mu[0])
		G_i[1].append(-1. * exp_mu[1])
		G_i[2].append(-1. * exp_mu[2])
		h_i.append(0)

		G_i[0]= G_i[0] + [0., -1., 0., 0.]
		G_i[1]= G_i[1] + [0., 0., -1., 0.]
		G_i[2]= G_i[2] + [0., 0., 0., -1.]
		h_i = h_i + [1., 0., 0., 0.]

		G = matrix(G_i)
		h = matrix(h_i)
		dims = {'l': len(mus)+1, 'q': [4], 's': []}
		sol = solvers.conelp(c, G, h, dims)
		sol['status']
		solution = np.array(sol['x'])
		if solution != None:
			solution=solution.reshape(3)
			w_delta_mu=np.dot(solution, exp_mu - mus[i])
			w_delta_mus.append(w_delta_mu)
		else:
			w_delta_mus.append(None)
		solutions.append(solution)
	solution = solutions[np.argmax(w_delta_mus)]
	w_delta_mu = w_delta_mus[np.argmax(w_delta_mus)]
	return solution, w_delta_mu

def optimal_feature(grids, steps, policy, epsilon = 1e-5, gamma= 0.5):
	exp_u= np.zeros(3)
	features= np.array(grids.features)
	if steps + 1 != steps:	##if step number is limited
		features_temp = np.array(grids.features)
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				action = int(policy[i, j])
				transition = np.array(grids.transitions[i, j, action])
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						features_temp[i, j] = features_temp[i, j] + np.multiply(transition[m, n], gamma * features[m, n])	
		features= np.array(features_temp)
	
	diff = float("inf")
	while diff > epsilon and steps + 1 == steps:	##if step number is inf, not limitated
		diff = 0.
		features_temp = np.array(grids.features)	
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				action = int(policy[i, j])
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



def optimal_value(grids, policy, steps, epsilon = 1e-5, gamma = 0.5):
	values = np.array(grids.rewards)

	if steps + 1 != steps:		##if step number is limited
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

	diff = float("inf")
	while diff > epsilon:		##if step number is not limited
		diff = 0.
		values_temp = np.zeros([grids.y_max, grids.x_max])
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				max_value = float("-inf")
				k = policy[i, j]
				transition_k = grids.transitions[i, j, k]
				reward_k = np.multiply(transition_k, gamma * values)
				value_k = 0.
				for m in range(grids.y_max):
					for n in range(grids.x_max):
						value_k+= reward_k[m, n]
				values_temp[i, j] = grids.rewards[i, j] + value_k
				new_diff = abs(values[i, j] - values_temp[i, j])
				if new_diff >  diff:
					diff = new_diff
		values = np.array(values + 0.1 * (values_temp - values))	
	return values	

def update_policy(grids, steps, epsilon= 1e-5, gamma=0.5):	
	policy=np.ones([grids.y_max, grids.x_max])
	values = optimal_value(grids, steps= steps-1, policy, epsilon=epsilon, gamma=gamma)	
	Q = np.zeros([grids.x_max, grids.y_max, 5])
	diff = True
	while diff is True:
		diff = False	
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
				if  np.argmax(Q[i, j]) != policy[i, j]:
					policy[i, j] = np.argmax(Q[i, j])
					diff = True
	
	return policy




def expert_train(grids, expert, agent, steps, epsilon=1e-5, iteration=100, gamma=0.5, start_theta= None):
	if start_theta == None:
		start_theta=np.random.randint(-100, 100, 3)
	theta=start_theta/np.linalg.norm(start_theta, ord=2)
	grids.w_features(theta)
	policies=[]
	policy= update_policy(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies.append(policy)
	print "initial expected feature is"
	print policy
	print optimal_feature(grids, steps, policy, epsilon = epsilon, gamma= gamma)

	#just add a random policy to policy set
	for i in range(iteration):
		
		new_theta, w_delta_mu = expert_update_theta(grids, expert, agent, steps, policies,  gamma)
		#calculate optimal weight for all policies and find out the optimal expected features witin the policy set 	 
		#calculate the expert expected weighted feature
		print i, " iteration", "old theta: ", theta, "weighted delta mu: ", w_delta_mu, "new theta: ", new_theta, 
		if abs(w_delta_mu) < epsilon:
			print "|expert_w_mu - w_mu| = ", abs(w_delta_mu), " < ", epsilon * 1e-2
			draw_grids(grids.rewards, None)
			return grids, theta, policies[-1]
		print "start learning...."
		theta = new_theta
		grids.w_features(theta)
		#if weighted weighted feature approximates the expert, end training
		new_policy = update_policy(grids, steps = steps, epsilon= epsilon)
		print "new policy generated...begin next iteration"
		policies.append(new_policy)
		print "new expected feature is"
		print optimal_feature(grids, steps, new_policy, epsilon = epsilon, gamma= gamma)
		#find optimal policy with new theta, 
		#add new policy to policies
	print "fail to converge, the current policy is:"
	print policies[-1]
	return grids, theta, policies[-1]
def expert_update_theta(grids, expert, agent, steps, policies, gamma=0.5, epsilon = 1e-5):
	mus=[]
	exp_mu = expert
	for policy in policies:
		mu = optimal_feature(grids, steps, policy, epsilon = epsilon, gamma=gamma)
		mus.append(mu)
	w_delta_mus=[]
	solutions=[]
	for i in range(len(mus)):
		G_i=[[], [], []]
		h_i = []
		c = matrix(mus[i] - exp_mu)
		for j in range(len(mus)):
			G_i[0].append(-1 * mus[i][0] - (-1) * mus[j][0])
			G_i[1].append(-1 * mus[i][1] - (-1) * mus[j][1])
			G_i[2].append(-1 * mus[i][2] - (-1) * mus[j][2])
			h_i.append(0)

		G_i[0]= G_i[0] + [0., -1., 0., 0.]
		G_i[1]= G_i[1] + [0., 0., -1., 0.]
		G_i[2]= G_i[2] + [0., 0., 0., -1.]
		h_i = h_i + [1., 0., 0., 0.]

		G = matrix(G_i)
	#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
		h = matrix(h_i)
		dims = {'l': len(mus), 'q': [4], 's': []}
		sol = solvers.conelp(c, G, h, dims)
		sol['status']
		solution = np.array(sol['x'])
		if solution != None:
			solution=solution.reshape(3)
			w_delta_mu=np.dot(solution, exp_mu - mus[i])
			w_delta_mus.append(w_delta_mu)
		else:
			w_delta_mus.append(None)
		solutions.append(solution)
	solution = solutions[np.argmax(w_delta_mus)]
	w_delta_mu = w_delta_mus[np.argmax(w_delta_mus)]
	return solution, w_delta_mu
		
	
class train:
	def __init__(self, grids= grid(12, 12, 0.6), start = None, steps = float("inf"), epsilon = 1e-2, gamma = 0.6, iteration = 30, theta = np.array([1./3., 1./3., -3./3.])):
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
		self.exp_mu =real_optimal(self.grids, self.agent, steps = self.steps, theta = self.theta, gamma=self.gamma, epsilon = self.epsilon)
		print self.grids.rewards
		print self.exp_mu
		self.real_rewards = np.array(self.grids.rewards)
		pylab.ion()
		pylab.title('Real Reward. Try real expert? Type [y/n] in the terminal')
		draw_grids(self.grids.rewards, None)
		pylab.ioff()
			

	def real_expert_train(self, epsilon= None):
		if epsilon is None:
			epsilon = self.epsilon
		pylab.ion()
		pylab.title('Rewards from expert train, close to continue')
		_, _, self.expert_policy= expert_train(self.grids, self.exp_mu, self.agent, steps= self.steps, epsilon= epsilon, iteration=self.iteration, gamma=self.gamma, start_theta= self.theta * (-1.0))
		pylab.ioff()
		file = open('expert_policy', 'w')
		for i in self.expert_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.close()
	
	
	def human_train(self, epsilon = None):
		if epsilon is None:
			epsilon = self.epsilon
		again = 'y'
		while(again != 'n' and again!= 'N'):
			if again != 'y' and again!= 'Y':
				print "Invalid input, exit...??"
				break
			else:
				try:
					a= input("Please input start point in [ , ] form, or click 'enter' to generate random point: ")
				except:
					a= np.random.randint(1, self.grids.x_max-2, 2)
				start=np.array(a)
				expert_temp, again = demo(self.grids, self.agent, start, steps= self.steps, gamma=self.gamma, epsilon= epsilon)
				self.expert.append(expert_temp)
	
		print "Start training..."
			
		demo_mu = np.zeros(3)
		for i in  range(len(self.expert)):
			demo_mu = demo_mu + self.expert[i]["mu"] 
		demo_mu = demo_mu/len(self.expert)
	
		print "expected demo mu is ", demo_mu
		_, _, self.demo_policy = expert_train(self.grids, demo_mu, self.agent, epsilon = epsilon, steps= self.steps, iteration = self.iteration, gamma=self.gamma, start_theta = demo_mu)
		draw_grids(self.grids.rewards, None)
		file = open('demo_policy', 'w')
		for i in self.demo_policy:
			for j in i:
				file.write(str(j)+":")
			file.write("\n")
		file.close()

	'''
	the demonstration always fail to converge, because the expected feature derived from it is just not big enough. If the analytical expected feature derived from the real reward function is shrinked by 10 times, the program never converges too. The demonstration also actually comes from different reward function and policy, it depends on how we thought each time.
	'''

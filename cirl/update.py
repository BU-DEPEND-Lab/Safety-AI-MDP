from car import car
from grid import grid

from cvxopt import matrix, solvers

import numpy as np
import matplotlib
import pylab




def demo(grids, agent, start, step=None, gamma=0.5):
	expert={}
	agent.state=grids.states[start[0], start[1]]
	theta=np.array([1./6., 1/3., 1./3.])
	theta=theta/np.linalg.norm(theta, ord=2)
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0], agent.state[1]]}]
	grids.w_features(theta)
	draw_grids(grids.rewards, trajectory)
	mu=np.zeros(3)
	while(1):
		action = input("Next action is ")
		if step == None and action == 0:
			break
		elif step != None:
			step = step - 1	
			if step == 0:
				break
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(action=action)})
		trajectory[-1]["feature"]=grids.features[trajectory[-1]["state"][0], trajectory[-1]["state"][1]]
		grids.w_features(theta)
		draw_grids(grids.rewards, trajectory)
	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	expert["mu"]=mu
	expert["trajectory"]=trajectory
 	playagain=raw_input("Want to play again? [y/n]?")
	return expert, playagain
	
def calc_u(grids, agent, policy, step, gamma=0.5):
	mu=np.zeros(3)
	trajectory=[{"state":agent.state, "feature": grids.features[agent.state[0], agent.state[1]]}]
	for i in range(step):
		action=policy[agent.state[0], agent.state[1]]
		trajectory[-1]["action"]=action
		trajectory.append({"state": agent.move(action)})
		trajectory[-1]["feature"]=grids.features[trajectory[-1]["state"][0], trajectory[-1]["state"][1]]
	for i in range(len(trajectory)):
		mu = mu + (gamma**i) * trajectory[i]["feature"]
	return mu, trajectory

def exp_u(grids, agent, policy, start, start_action=None, step=None, epoch=1000, gamma=0.5):
	if step == None:
		step = epoch
	mu=np.zeros([3])
	agent.state=np.array([start[0], start[1]])
	trajectory_i_j={}
	for i in range(epoch):
		if start_action != None:
			org_action=policy[agent.state[0], agent.state[1]]
			policy[agent.state[0], agent.state[1]]=start_action
			mu_i_j_1, _= calc_u(grids, agent, policy , step=1)
			policy[agent.state[0], agent.state[1]]=org_action
		else:
			mu_i_j_1 = np.zeros([3])
		mu_i_j, _ =calc_u(grids, agent, policy, step=step)
		mu = mu + mu_i_j + mu_i_j_1
		#	draw(rewards, trajectory)
	mu=mu/epoch
	return mu


def draw_grids(rewards, trajectory):
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
	pylab.axis([0,10, 10,0])
	pylab.savefig('plt.png')
	pylab.show()

def update_theta(grids, agent, policies, start, theta=None, step=None, epoch=1000, penalty=-0.5, gamma=0.5):
	mus=[]
	if theta == None:
		theta=np.random.randint(-10.0, 10.0, 3)
		theta=theta/np.linalg.norm(theta, ord=2)
	for policy in policies:
		mu = exp_u(grids=grids, agent=agent, policy=policy, start=start, step=None, epoch=epoch, start_action=None)
		mus.append(mu)
	w_mus=[]
	solutions=[]
	for i in range(len(mus)):
		mus_i=[[], [], []]
		h_i = []
		c = matrix(-1 * mus[i])
		for j in range(len(mus)):
			mus_i[0].append(mus[i][0] - mus[j][0])
			mus_i[1].append(mus[i][1] - mus[j][1])
			mus_i[2].append(mus[i][2] - mus[j][2])
			h_i.append(0)
		mus_i[0]= mus_i[0] + [0., -1., 0., 0.]
		mus_i[1]= mus_i[1] + [0., 0., -1., 0.]
		mus_i[2]= mus_i[2] + [0., 0., 0., -1.]
		h_i = h_i + [1., 0., 0., 0.]
		print h_i, mus_i
	#	G = matrix([[-1.0 * mus[i][0], 0., -1., 0., 0.], [-1. * mus[i][1], 0., 0., -1., 0.], [-1 * mus[i][2], 0., 0., 0., -1.]])

		G = matrix(mus_i)
	#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
		h = matrix(h_i)
		dims = {'l': len(mus), 'q': [4], 's': []}
		sol = solvers.conelp(c, G, h, dims)
		sol['status']
		solution = np.array(sol['x']).reshape(3)
		if solution != None:
			w_mu=np.dot(solution, mus[i])
			w_mus.append(w_mu)
		else:
			w_mus.append(None)
		solutions.append(solution)
	solution = solutions[np.argmin(w_mus)]
	w_mu = w_mus[np.argmin(w_mus)]
#	if w_mu < 0:
#		solution = -1 * solution / np.linalg.norm(solution, ord=2)
#		w_mu = -1 * w_mu / np.linalg.norm(solution, ord=2)
#	elif w_mu > 0:
#		solution = solution / np.linalg.norm(solution, ord=2)
#		w_mu = w_mu / np.linalg.norm(solution, ord=2)
	return solution, w_mu

def optimal_value(grids, epsilon = 1e-2, gamma = 0.5):
	values = grids.rewards
	diff = float("inf")
	while diff > epsilon:
		diff = 0
		for i in range(grids.y_max):
			for j in range(grids.x_max):
				max_value = float("-inf")
				for k in range(5):
					value_k=0
					transition_k = grids.transitions[i, j, k]
					reward_k = np.dot(transition_k, grids.rewards + gamma * values)
					for i in range(len(grids.states)):
						for j in range(len(grids.states[i])):
							value_k+= reward_k[i, j]
					max_value = max(value_k, max_value)		
				new_diff = abs(values[i, j] - max_value)
				if new_diff > diff:
					diff = new_diff
				values[i, j] = max_value
	return values	

def update_policy(grids, epsilon=0.01, gamma=0.5):	
	policies=np.ones([grids.y_max, grids.x_max])
	values = optimal_value(grids, epsilon=1e-2, gamma=0.5)
	Q = np.zeros([grids.x_max, grids.y_max, 5])
	for i in range(grids.y_max):
		for j in range(grids.x_max):
			for k in range(5):
				value_k=0 
				transition_k = grids.transitions[i, j, k]
				reward_k = np.dot(transition_k, grids.rewards + gamma * values)
				for i in range(len(grids.states)):
					for j in range(len(grids.states[i])):
						value_k+= reward_k[i, j]
				Q[i, j, k] = value_k
		##Q -= Q.max(axis=2).reshape((grids.y_max, grids.x_max, 1))
		##Q = np.exp(Q)/np.exp(Q).sum(axis=2)
				policies[i, j] = np.argmax(Q[i, j])
	return policies



def train(grids, expert, agent, epsilon=0.1, iteration=100, epoch=1000, gamma=0.5):
	step = len(expert["trajectory"])
	start = np.array([expert["trajectory"][0]["state"][0], expert["trajectory"][0]["state"][1]])
	theta=np.random.randint(-100, 100, 3)
	theta=theta/np.linalg.norm(theta, ord=2)
	policies=[]
	#just add a random policy to policy set
	random_policy = np.random.randint(0, 5, [grids.y_max, grids.x_max])
	policies.append(random_policy)
	for i in range(iteration):
		theta, w_mu = update_theta(grids, agent, policies, start, theta, step, epoch, penalty=-10)
		#calculate optimal weight for all policies and find out the optimal expected features witin the policy set 	 
		expert_w_mu = np.dot(theta, expert["mu"])
		#calculate the expert expected weighted feature
		print i, " iteration", "theta", theta, "weighted mu:", w_mu, "expert weighted mu:", expert_w_mu
		if abs(expert_w_mu - w_mu) < epsilon:
			print "expert_w_mu - w_mu =", expert_w_mu -w_mu, " < ", epsilon
			grids.w_features(theta)
			draw_grids(grids.rewards, None)
			return grids
		print "start learning...."
		grids.w_features(theta)
		#if weighted weighted feature approximates the expert, end training
		new_policy = update_policy(grids, epsilon=0.01)
		policies.append(new_policy)
		print "new policy generated...begin next iteration"
		#find optimal policy with new theta, 
		#add new policy to policies
	print "fail to converge"
	return grids
		
	
if __name__=="__main__":
	grids=grid()
	agent=car(states=grids.states)
	policies=[]
	policies.append(np.ones([grids.y_max, grids.x_max]))
	start=np.array([5, 5])
	expert, again=demo(grids, agent, start, gamma=0.5)
	if again != 'n' and again!= 'N':
		if again != 'y' and again!= 'Y':
			print "What are you talking about??"
		else:
			expert, again = demo(grids, agent, start, gamma=0.5)
	while again=='y' or again=='Y':
		expert, again = demo(grids, agent, start, gamma=0.5)
	if again == 'n' or again == 'N':
		print "Start training...."
		grids=train(grids, expert, agent, 0.01, iteration=10000000, gamma=0.5)
	print grids.rewards
	draw_grids(grids.rewards, None)

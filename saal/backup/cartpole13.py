import gym
import re
import cProfile
import math
from grid_v2 import grid
from car import car
import update
import numpy as np
import util

# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.
import MDP

# Number of distretized pieces for each observation component
coords = [8, 4, 4, 4];
steps = 200
gamma = 0.9
epsilon = 1e-3
order = 5
iteration = 20
# We disregard the cart position on the screen to make learning
# faster
threshes = [0.3, 0.4, 0.4, 0.4];
maxepisodes = 5000

# Gym parameters
render = 0;
record = 0;
recordfolder = './cartpole-experiment-4'

env = gym.make('CartPole-v0')

# Action space is 2, State space depends on coords
A = env.action_space.n;
print A
#S = coords**4 * env.observation_space.shape[0];
#S = env.observation_space.shape[0] 
S = 1
for coord in coords:
	S *= coord
print S 

grids = grid(int(coords[0]), int(S/coords[0]), A, probability = 1.0, sink = False)
grids.transitions = np.float64(np.zeros([grids.y_max, grids.x_max, A, grids.y_max, grids.x_max]))

f = 16
grids.features = np.zeros([grids.y_max, grids.x_max, 4 + f])
feature_states = []
#feature_states.append([])
#feature_states.append([])
for i in range(f):
	s = int(S * i/f)
	feature_states.append([int(s/coords[0]), int(s%coords[0])])
	'''
	if i%2 == 0:
  		feature_states[0].append([int(s/coords[0]), int(s%coords[0])])
	else
		feature_states[1].append([int(s/coords[0]), int(s%coords[0])])
	'''
'''	
print feature_states

for i in range(grids.y_max):
	for j in range(grids.x_max):
		index = 0
		for l in range(0, len(feature_states)):
			for m in range(0, len(feature_states[l])):
				grids.features[i, j, index + m] = math.exp(-0.25 * math.sqrt((i - feature_states[l][m][0])**2 + (j - feature_states[l][m][1])**2))
			index  = index + len(feature_states[l])
'''

for i in range(grids.y_max):
	for j in range(grids.x_max):
		grids.features[i, j, 0] = math.exp(- coords[0] * 1.0 * j/coords[0])
		grids.features[i, j, 1] = math.exp(- coords[1] * 1.0 * (i%coords[1])/coords[1])
		grids.features[i, j, 2] = math.exp(- coords[2] * 1.0 * ((i/coords[1])%coords[2])/coords[2])
		grids.features[i, j, 3] = math.exp(- coords[3] * ((1.0 * i/coords[1])/coords[2])/coords[3])
		for k in range(f):
			grids.features[i, j, k + 4] = math.exp(-0.25 * math.sqrt((1.0 * i - feature_states[k][0])**2 + (1.0 * j - feature_states[k][1])**2))

theta = np.ones([len(grids.features[-1][-1])])
theta = theta/np.linalg.norm(theta, ord = 2)

unsafe = []
for s in range(S):
	if (int(s%coords[0]) <= 1 and int(((s/coords[0])/coords[1])%coords[2]) <=  0)  or (int(s%coords[0]) >= (coords[0] - 2) and int(((s/coords[0])/coords[1])%coords[2]) >= coords[2] - 1):
		unsafe.append([int(s/coords[0]), int(s%coords[0])])


safety = 0.01

agent = car(states=grids.states)

starts = []
# We are not going to assume anything here. We are just going to
# approximate the observation space in a finite number of states.
# In particular, we approximate each vector component in 4 coords.
# If we discard the first component (the cart position on the
# screen) we can learn faster, but adding it still works.
# Then we use PrioritizedSweeping in order to extract as much
# information as possible from each datapoint. Finally we select
# actions using a softmax policy.


exp = MDP.SparseExperience(S, A);
model = MDP.SparseRLModel(exp, gamma);
solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
policy = MDP.QGreedyPolicy(solver.getQFunction());

def observationToState(o, thresh):
    s = int(0);
    exp = 1;
    for i in range(len(o)):
        if thresh[i] == 0.0:
            continue
	elif i == 0:
		ox = (min(thresh[i], max(-thresh[i], o[i])) + thresh[i]);
		if ox <= 0.0:
			val = 0
		elif ox >= 2 * thresh[0]:
			val = coords[0] - 1
		else:
			for j in range(1, coords[0] - 1):
				if ox < float(j * 2.0 * thresh[i] / (coords[0] - 2)):
					val = j
					break
	else:
        	ox = (min(thresh[i], max(-thresh[i], o[i])) + thresh[i]);
	## if o[i] is in [-thresh[i], thresh[i]], then ox=o[i] + thresh[i]
	## if o[i] < -thresh[i], then ox = 0
	## if o[i] > thresh[i], then ox = 2*thresh[i]
        	val = int((ox * float(coords[i])) / (2.0 * thresh[i]));
	## val <= coords * (ox/(2*thresh[i])), ox/(2*thresh[i]) is in [0, 1]
	## val is the state index in [0, 1, 2, 3]
	## so if o[i] < -thresh[i], val = 0; if o[i] > thresh[i], val = 3
	## if val == coords, means that ox = 2*thresh[i]
	## but we start from 0, val=$coords - 1 = 3
	## thus val is in {0, 1, 2, 3}
        if val == coords[i]:
            val = coords[i] - 1
	s += val * exp;
        exp *= coords[i];
    return s;

if record:
    env.monitor.start(recordfolder)

# We use the following two variables to track how we are doing.
# Read more at the bottom
transitions = np.zeros([S, A, S]).astype(float)


demo_mu = np.zeros(len(grids.features[-1][-1]))
using = 0
real = raw_input('Setting up MDP? [Y/N]')
while real == 'Y' or real == 'y':
	#demo_mu_episode = np.zeros(len(grids.features[-1][-1]))
	#demo_mu_episodes = np.zeros(len(grids.features[-1][-1]))

	episodes=0
	win = 0
	streak = list()
	
	for i_episode in xrange(maxepisodes):
    		o = env.reset()
		start_s = observationToState(o, threshes)
		add_start = True
		dead = False
		#starts = [grids.states[int(start_s/grids.x_max)][int(start_s%grids.x_max)]]
		for start in starts:
			if start[0] == int(start_s/grids.x_max) and start[1] == int(start_s%grids.x_max):
				add_start = False
		if add_start:
			starts.append(grids.states[int(start_s/grids.x_max)][int(start_s%grids.x_max)])
			
    		demo_mu_episode = np.zeros(len(grids.features[-1][-1]))
    		for t in xrange(steps):
        	#	if render or i_episode == maxepisodes - 1:
        	 #   		env.render()

        		# Convert the observation into our own space
        		s = observationToState(o, threshes);
        		# Select the best action according to the policy
        		a = policy.sampleAction(s)
        		# Act
        		o1, rew, done, info = env.step(a);
        		# See where we arrived
        		s1 = observationToState(o1, threshes);

			transitions[s, a, s1] += 1.0

			#if int(s1%coords[0]) == 0 or int(s1%coords[0]) == coords[0] - 1:
			#	done = True
			for s_ in unsafe:
				if s_[0] * coords[0] + s_[1] == s1:
					dead = True

        		if done:
            			break;

        # Record information, and then run PrioritizedSweeping
        		exp.record(s, a, s1, rew);
       		 	model.sync(s, a, s1);
		        solver.stepUpdateQ(s, a);
		        solver.batchUpdateQ();
		    	#demo_mu += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
		    	demo_mu_episode += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
  		  	#demo_mu_episodes += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)

		        o = o1;

   		#	if render or i_episode == maxepisodes - 1:
      		#		env.render()

    		tag = '   ';
    # Here we have to set the reward since otherwise rewards are
    # always 1.0, so there would be no way for the agent to distinguish
    # between bad actions and good actions.
    		if done:
        		rew = -100;


    		if t >= steps - 1:
       			tag = '###';
      			win += 1;
        		streak.append(1)
			if dead is False:
				using += 1;
				demo_mu += demo_mu_episode
  		else:
		        streak.append(0)
			'''
			for tt in range(t + 1, steps):
				demo_mu += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
		    		demo_mu_episode += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
  		  		demo_mu_episodes += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)

			add = True
			for b in unsafe:
				if b[0] * coords[0] + b[1] == s1:
					add = False
					break
			if add:
				unsafe.append([int(s1/coords[0]), int(s1%coords[0])])
			'''
  		if len(streak) > 100:
      			streak.pop(0)

    		episodes +=1;
    		#using += 1;
    		exp.record(s, a, s1, rew);
    		model.sync(s, a, s1);
    		solver.stepUpdateQ(s, a);
    		solver.batchUpdateQ();
    		#demo_mu += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
    		#demo_mu_episode += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
    		#demo_mu_episodes += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
    # If the learning process gets stuck in some local optima without
    # winning we just reset the learning. We don't want to try to change
    # what the agent has learned because this task is very easy to fail
    # when trying to learn something new (simple exploration will probably
    # just make the pole topple over). We just want to learn the correct
    # thing once and be done with it.
		if episodes == 100:
    			#if sum(streak) < 90:
			#	using -= episodes
			#	demo_mu -= demo_mu_episodes 
			#else:
			#	for s_ in unsafe:
			#		if s_[0] * coords[0] + s_[1] == s1:
			#			using -= episodes
			#			demo_mu -= demo_mu_episodes 
    			if sum(streak) < 30:
	        		exp = MDP.SparseExperience(S, A);
       				model = MDP.SparseRLModel(exp, gamma);
        			solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
      	  			policy = MDP.QGreedyPolicy(solver.getQFunction());
			demo_mu_episodes = np.zeros([len(grids.features[-1][-1])])
			episodes = 0

	    	print "Episode {} finished after {} timecoords {} {} {}".format(i_episode, t+1, tag, win, using)

	

	file = open('cartpole_MDP', 'w')
	file.write('starts\n')
	for start in starts:
		file.write(str(int(start[0]))+':'+str(int(start[1]))+'\n')

	demo_mu /= using
	print demo_mu
	file.write('features\n')
	for feature in demo_mu:
		file.write(str(feature)+':')
	file.write('\n')

	file.write('transitions\n')
	grids.transitions = np.float64(np.zeros([grids.y_max, grids.x_max, A, grids.y_max, grids.x_max]))
	for s in range(S):
		for a in range(A):
			transitions_ = np.float64(np.zeros([S]))
			if np.sum(transitions[s, a]) > 0.0:
				transitions_ = util.vector_normalizer(transitions[s, a], order = order, anchor = s)
				#transitions_ = np.round(transitions[s, a]/np.sum(transitions[s, a]), order)
				#transitions_ = transitions[s, a] / np.sum(transitions[s, a])
			elif False:
				for s_ in range(S):
					transitions_[s_] = np.round(model.getTransitionProbability(s, a, s_), order)
				#transitions_ = transitions_ / np.sum(transitions)
				transitions_ = util.vector_normalizer(transitions_, order = order, anchor = s)
			p_sum = np.float64([0.0])
			for s_ in range(S):
				p = np.float64([transitions_[s_]])
				grids.transitions[int(s/coords[0]), int(s%coords[0]), a, int(s_/coords[0]), int(s_%coords[0])] = p[0]
				if p >= 0.0:
					file.write(str(int(s/coords[0]))+str(':')+str(int(s%coords[0]))+str(':')+str(a)+str(':')+str(int(s_/coords[0]))+str(':')+str(int(s_%coords[0]))+str(':')+str(p[0]))
					file.write('\n')
				p_sum += p
		
			'''
			if p_sum[0] > 1.0:	
				print "Still p_sum=", p_sum[0], "> 1.0 hehehehe"
				grids.transitions[int(s/coords[0]), int(s%coords[0]), a] /= np.sum(grids.transitions[int(s/coords[0]), int(s%coords[0]), a])
			elif p_sum[0] < 1.0:
				print "Still p_sum=", p_sum[0], "< 1.0 hahahaha"
				grids.transitions[int(s/coords[0]), int(s%coords[0]), a, int(s/coords[0]), int(s%coords[0])] += (np.float64([1.0]) - p_sum)[0]
				grids.transitions[int(s/coords[0]), int(s%coords[0]), a] /= np.sum(grids.transitions[int(s/coords[0]), int(s%coords[0]), a])
			'''
					
	file.close()

	file = open('cartpole_unsafe', 'w')
	for b in unsafe:
		file.write(str(b[0]) + ':' + str(b[1]) + '\n')
	file.close()

	file = open('cartpole_expert_policy', 'w')
	for i in range(len(demo_mu)):
		file.write(str(demo_mu[i])+':')
	file.write('\n')
	for i in range(len(theta)):
		file.write(str(':'))
	file.write('\n')
	for i in range(grids.y_max):
		for j in range(grids.x_max):
			file.write(str(policy.sampleAction(i * grids.x_max + j)) + str(':'))
  		file.write('\n')
	file.close()

	if record:
    		env.monitor.close()
	real = raw_input('Setting up MDP? [Y/N]')


real=raw_input("Try Apprenticeship Learning? [Y/N]")
if real == 'Y' or real == 'y':
	print 'Yeah!!!'
	real=raw_input("Human demonstrate? [Y/N]")
	while real == 'Y' or real == 'y':
		demo_mu = np.zeros(len(grids.features[-1][-1]))
		done = False
		diff = -float('inf')
		o = env.reset()
		for t in xrange(steps):
			demo_mu_temp = np.array(demo_mu)
			s =  observationToState(o, threshes); 
			demo_mu += grids.features[int(s/coords[0])][int(s%coords[0])] * (gamma**t)
			diff = np.linalg.norm(demo_mu - demo_mu_temp, ord = 2 )
		    	if diff < 1e-5 and done is True:
				break;
	
       		 	# Convert the observation into our own space
	      	  	# Select the best action according to the policy
	        	# Act
			if done is False:
	        		a = raw_input("Next step is ")
				a = int(a)
	       	 	o1, rew, done, info = env.step(a);
	        	# See where we arrived
	        	s1 = observationToState(o1, threshes);


	        	# Record information, and then run PrioritizedSweeping
	        	exp.record(s, a, s1, rew);
	        	model.sync(s, a, s1);
	  	      	solver.stepUpdateQ(s, a);
	 	       	solver.batchUpdateQ();

	        	o = o1;

	    		env.render()
		print "Episode {} finished after {} timecoords {}".format(i_episode, t+1, win)
		print "Expert feature count ", demo_mu
		if record:
	    		env.monitor.close()
		real=raw_input("Human demonstrate again? [Y/N]")
	
	
	file = open('cartpole_MDP', 'r')
	lines = file.readlines()
	if lines[0] == 'starts' or lines[0] == 'starts\n':
		index = 1
	else:
		exit()
	starts = []
	for index in range(1, len(lines)):
		if lines[index] != 'features' and lines[index] != 'features\n':
			line = re.split(':|\n|''', lines[index])
			starts.append([int(line[0]), int(line[1])])
		else:
			break
	index += 1	

	demo_mu = []
	line = lines[index].split(':')
	for feature in line:
		try:
			demo_mu.append(float(feature))
		except:			
			continue
	demo_mu = np.array(demo_mu)
	print demo_mu
	index += 2

	for i in range(index, len(lines)):
		line = re.split(':|\n|''', lines[i])
		grids.transitions[int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])] = np.float64([line[5]])[0]
	file.close()

	#cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe, toolbox)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': starts, 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': True, 'safety': None, 'unsafe': unsafe, 'toolbox': False, 'test_expert_train': update.expert_train}, {})
	#cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': starts, 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': False, 'safety': None, 'unsafe': unsafe, 'toolbox': False, 'test_expert_train': update.expert_train}, {})
#	cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': starts, 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': False, 'safety': None, 'unsafe': unsafe, 'toolbox': True, 'test_expert_train': update.expert_train}, {})
	_, theta, policy ,_ = update.expert_train(grids, demo_mu, agent, epsilon = epsilon, starts = starts, steps = steps, iteration = 30, gamma= gamma, start_theta = np.array(demo_mu)/np.linalg.norm(demo_mu, ord=2), MC = False, safety = None, unsafe = unsafe)
	grids.w_features(theta)

	file = open('cartpole_policy', 'w')
	for i in range(len(demo_mu)):
		file.write(str(demo_mu[i])+':')
	file.write('\n')
	for i in range(len(theta)):
		file.write(str(theta[i])+':')
	file.write('\n')
	for i in policy:
		for j in i:
			file.write(str(j) + str(':'))
  		file.write('\n')
	file.close()

real = raw_input("Try policy? [Y/N]")	
while real == 'y' or real == 'Y':
	file = open('cartpole_policy', 'r')
	lines = file.readlines()

	demo_mu_str = re.split(':|\n|''', lines[0])
	demo_mu = []
	for i in demo_mu_str:
		if i != '' and i!=']' and i!= ' ' and i!='\n':
			demo_mu.append(float(i))
	demo_mu = np.array(demo_mu)
	print demo_mu

	theta_str = re.split(':|\n|''', lines[1])
	theta = []
	for i in theta_str:
		if i != '' and i!=']' and i!= ' ' and i!='\n':
			theta.append(float(i))
	theta = np.array(theta)
	theta = theta/np.linalg.norm(theta, ord= 2)
	print theta

	policy = np.zeros([grids.y_max, grids.x_max])
	for i in range(2, len(lines)):
		line = re.split(':|\n|''', lines[i])
		j = 0
		for k in range(len(line)):
			if line[k] != '':
				policy[i-2, j] = float(line[k])
				j += 1
	file.close()
	o = env.reset()
	t = 0
	s =  observationToState(o, threshes); 
	print "initial state ", int(s/coords[0]), ", ", int(s%coords[0])
	for t in xrange(steps):
		s =  observationToState(o, threshes); 
     		   # Convert the observation into our own space
      	 	 # Select the best action according to the policy
     		a = int(policy[int(s/coords[0]), int(s%coords[0])])
       		# Act
      	  	o1, rew, done, info = env.step(a);
		if done:
			break
      	        # See where we arrived
     		s1 = observationToState(o1, threshes);

     	   	# Record information, and then run PrioritizedSweeping
        	exp.record(s, a, s1, rew);
        	model.sync(s, a, s1);
        	solver.stepUpdateQ(s, a);
        	solver.batchUpdateQ();

        	o = o1;

    		#env.render()
	if record:
    		env.monitor.close()
	print t
	real = raw_input("Try policy? [Y/N]")	

real=raw_input("Try synthesizing? [Y/N]")
while real == 'Y' or real == 'y':

	file = open('cartpole_policy', 'r')
	lines = file.readlines()

	demo_mu_str = re.split(':|\n|''', lines[0])
	demo_mu = []
	for i in demo_mu_str:
		if i != '' and i!=']' and i!= ' ' and i!='\n':
			demo_mu.append(float(i))
	demo_mu = np.array(demo_mu)
	print demo_mu

	theta_str = re.split(':|\n|''', lines[1])
	theta = []
	for i in theta_str:
		if i != '' and i!=']' and i!= ' ' and i!='\n':
			theta.append(float(i))
	theta = np.array(theta)
	theta = theta/np.linalg.norm(theta, ord= 2)
	print theta

	al_policy = np.zeros([grids.y_max, grids.x_max]).astype(float)
	for i in range(2, len(lines)):
		line = re.split(':|\n|''', lines[i])
		j = 0
		for k in range(len(line)):
			if line[k] != '':
				al_policy[i-2, j] = float(line[k])
				j += 1
	file.close()

	file = open('cartpole_expert_policy', 'r')
	lines = file.readlines()

	
	exp_policy = np.zeros([grids.y_max, grids.x_max]).astype(float)
	try:
		for i in range(2, len(lines)):
			line = re.split(':|\n|''', lines[i])
			j = 0
			for k in range(len(line)):
				if line[k] != '':
					exp_policy[i-2, j] = float(line[k])
					j += 1
		file.close()
	except:
		exp_policy = None

	file = open('cartpole_MDP', 'r')
	lines = file.readlines()
	if lines[0] == 'starts' or lines[0] == 'starts\n':
		index = 1
	else:
		exit()
	starts = []
	for index in range(1, len(lines)):
		if lines[index] != 'features' and lines[index] != 'features\n':
			line = re.split(':|\n|''', lines[index])
			starts.append([int(line[0]), int(line[1])])
		else:
			break
	index += 1	

	demo_mu = []
	line = lines[index].split(':')
	for feature in line:
		try:
			demo_mu.append(float(feature))
		except:			
			continue
	demo_mu = np.array(demo_mu)
	index += 2

	for i in range(index, len(lines)):
		line = re.split(':|\n|''', lines[i])
		grids.transitions[int(line[0]), int(line[1]), int(line[2]), int(line[3]), int(line[4])] = np.float64([line[5]])[0]
	file.close()
		
	#unsafe = []
	#file = open('cartpole_unsafe', 'r')
	#lines = file.readlines()
	#for i in range(0, len(lines)):
	#	line = re.split(':|\n|''', lines[i])
	#	unsafe.append([int(line[0]), int(line[1])])
	#file.close()	
	
	p_B_sum = update.output_model(grids, starts, al_policy, steps, unsafe, 0.0)
	print "Model checking unsafe prob ", p_B_sum
	'''
	exp_u_G, p_G, exp_u_B, p_B, _ = update.sample_feature(grids, agent, starts, steps, al_policy, epochs= 5000, epsilon = epsilon, gamma=gamma, bounds = np.zeros([len(starts)]), unsafe = unsafe)
	new_mu = np.sum(np.reshape(exp_u_G, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_G, [grids.y_max*grids.x_max, 1]) +   np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]) * np.reshape(p_B, [grids.y_max*grids.x_max, 1]), 0) 
	p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
	print "AL policy unsafe rate ", p_B_sum
	'''
	safety_spec = raw_input("Safety spec? [0.0, 1.0]")
	try:
		safety = float(safety_spec)
	except:	
		break
	_, theta, policy, p = update.expert_synthesize(grids, demo_mu, agent, starts, steps = steps, epsilon = epsilon, iteration = iteration, gamma = gamma, start_theta = theta, MC = False, unsafe =unsafe, safety = safety, exp_policy = exp_policy)
	#grids.w_features(theta)
	#learnt_policy, value = update.optimal_value(grids, agent, steps = steps, epsilon= 1e-5, gamma = gamma, toolbox = False, starts = starts)
	file = open('cartpole_policy_safe', 'w')
	file.write(str(safety))
	file.write('\n')
	file.write(str(p))
	file.write('\n')
	for i in range(len(theta)):
		file.write(str(theta[i])+':')
	file.write('\n')
	for i in policy:
		for j in i:
			file.write(str(j) + str(':'))
  		file.write('\n')
	file.close()


real = raw_input("Try safe policy? [Y/N]")
while real == 'Y' or real == 'y':
	file = open('cartpole_policy_safe', 'r')
	lines = file.readlines()

	theta_str = re.split(':|\n|''', lines[1])
	theta = []
	for i in theta_str:
		if i != '' and i!=']' and i!= ' ' and i!='\n':
			theta.append(float(i))
	theta = np.array(theta)
	theta = theta/np.linalg.norm(theta, ord= 2)
	print theta

	policy = np.zeros([grids.y_max, grids.x_max])
	for i in range(3, len(lines)):
		line = re.split(':|\n|''', lines[i])
		j = 0
		for k in range(len(line)):
			if line[k] != '':
				policy[i-3, j] = float(line[k])
				j += 1
	file.close()

	o = env.reset()
	s =  observationToState(o, threshes); 
	print "initial state ", int(s/coords[0]), ", ", int(s%coords[0])
	t = 0
	for t in xrange(steps):
		s =  observationToState(o, threshes); 
     		   # Convert the observation into our own space
      	 	 # Select the best action according to the policy
     		a = int(policy[int(s/coords[0]), int(s%coords[0])])
       		# Act
      	  	o1, rew, done, info = env.step(a);
      	        # See where we arrived
     		s1 = observationToState(o1, threshes);

		if done:
			break
     	   	# Record information, and then run PrioritizedSweeping
        	exp.record(s, a, s1, rew);
        	model.sync(s, a, s1);
        	solver.stepUpdateQ(s, a);
        	solver.batchUpdateQ();

        	o = o1;

    		#env.render()
	if record:
    		env.monitor.close()
	print t
	real = raw_input("Try safe policy? [Y/N]")


real = raw_input("Compare policies? [Y/N]")
while real == 'Y' or real == 'y':
	file = open('cartpole_policy', 'r')
	lines = file.readlines()
	policy = np.zeros([grids.y_max, grids.x_max])
	for i in range(2, len(lines)):
		line = re.split(':|\n|''', lines[i])
		j = 0
		for k in range(len(line)):
			if line[k] != '':
				policy[i-2, j] = float(line[k])
				j += 1
	file.close()

	file = open('cartpole_policy_safe', 'r')
	lines = file.readlines()
	policy_ = np.zeros([grids.y_max, grids.x_max])
	for i in range(3, len(lines)):
		line = re.split(':|\n|''', lines[i])
		j = 0
		for k in range(len(line)):
			if line[k] != '':
				policy_[i-3, j] = float(line[k])
				j += 1
	file.close()

	average = 0.0
	average_ = 0.0
	dead_rate = 0.0
	dead_rate_ = 0.0
	total_score = 0.0
	total_dead = 0.0
	maxepisodes = 1000
	for i_episode in xrange(2 * maxepisodes):
		if i_episode == maxepisodes:
			average = total_score/maxepisodes
			dead_rate = total_dead
			total_score = 0.0
			total_dead = 0.0
			policy = np.array(policy_)
		o = env.reset()
		dead = False
		t = 0
		for t in xrange(steps):
			s =  observationToState(o, threshes); 
     		   	# Convert the observation into our own space
      	 	 	# Select the best action according to the policy
     			a = int(policy[int(s/coords[0]), int(s%coords[0])])
       			# Act
      	  		o1, rew, done, info = env.step(a);
      	        	# See where we arrived
     			s1 = observationToState(o1, threshes);

			#if int(s1%coords[0]) == 0 or int(s1%coords[0]) == coords[0] - 1:
			for s_ in unsafe:
				if s_[0] * coords[0] + s_[1] == s1:
					dead = True
			if done:
				break
     	   		# Record information, and then run PrioritizedSweeping

        		o = o1;
		total_score += t+1
		if dead:
			total_dead += 1
		print "Episode {} finished after {} timecoords, dead? {}, total dead {}".format(i_episode, t+1, dead, total_dead)
			
	
	average_ = total_score/maxepisodes
	dead_rate_ = total_dead

	print "policy average socre {}, dead times {}, rate {}".format(average, dead_rate, dead_rate/maxepisodes)
	print "policy_safe average socre {}, dead times {}, rate {}".format(average_, dead_rate_, dead_rate_/maxepisodes)
	
	real = raw_input("Compare policies? [Y/N]")

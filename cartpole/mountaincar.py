import gym
import cProfile
import math
from grid_v2 import grid
from update import train
from car import car
import update
import numpy as np
# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.
import MDP

# Number of distretized pieces for each observation component
coords = [37, 29]
# We disregard the cart position on the screen to make learning
# faster
threshes = [1.8/(coords[0]-1), 1.4/(coords[1]-1)];
maxepisodes = 5000 
combo = 0
steps = int(200/(combo+1))
# Gym parameters
render = 0;
record = 0;
recordfolder = './mountaincar'

gamma = 0.9

env = gym.make('MountainCar-v0')
print "hehe"

# Action space is 2, State space depends on coords
A = env.action_space.n;
#S = env.observation_space.shape[0] * coords * coords;
S = 10
for coord in coords:
	S = S * coord

# We are not going to assume anything here. We are just going to
# approximate the observation space in a finite number of states.
# In particular, we approximate each vector component in 4 coords.
# If we discard the first component (the cart position on the
# screen) we can learn faster, but adding it still works.
# Then we use PrioritizedSweeping in order to extract as much
# information as possible from each datapoint. Finally we select
# actions using a softmax policy.

exp = MDP.Experience(S, A);
model = MDP.RLModel(exp, gamma);
solver = MDP.PrioritizedSweepingRLModel(model, 0.1, 500);
policy = MDP.QGreedyPolicy(solver.getQFunction());

def observationToState(o, thresh): 
	x = int(1.2/thresh[0]) + int(o[0]/thresh[0])
	y = int(0.07/thresh[1]) + int(o[1]/thresh[1])
	s = y * coords[0] + x 
	return s


win = 0
# We use the following two variables to track how we are doing.
# Read more at the bottom
episodes=0
streak = list()
for i_episode in xrange(maxepisodes):
    o = env.reset()
    using = 0
    for t in xrange(steps):
        if render or i_episode == maxepisodes - 1:
         	env.render()

        	# Convert the observation into our own space
    	s = observationToState(o, threshes);
        	# Select the best action according to the policy
  	a = policy.sampleAction(s)
        	# Act
	for i in range(combo):
		env.step(a)
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

    
    if render or i_episode == maxepisodes - 1:
        env.render()

	
    if t < steps-1:
	win += 1
	streak.append(1)
	rew = 100
    else:
	streak.append(0)

    if len(streak) > 20:
	streak.pop(0)
    # Here we have to set the reward since otherwise rewards are
    # always 1.0, so there would be no way for the agent to distinguish
    # between bad actions and good actions.
    episodes += 1
    exp.record(s, a, s1, rew);
    model.sync(s, a, s1);
    solver.stepUpdateQ(s, a);
    solver.batchUpdateQ();

    # If the learning process gets stuck in some local optima without
    # winning we just reset the learning. We don't want to try to change
    # what the agent has learned because this task is very easy to fail
    # when trying to learn something new (simple exploration will probably
    # just make the pole topple over). We just want to learn the correct
    # thing once and be done with it
    if sum(streak) < 10 and episodes > 300:
        episodes = 0
	exp = MDP.Experience(S, A);
	model = MDP.RLModel(exp, gamma);
	solver = MDP.PrioritizedSweepingRLModel(model, 0.1, 500);
	policy = MDP.QGreedyPolicy(solver.getQFunction());

    print "Episode {} finished after {} timecoords {} {}".format(i_episode, t+1, win, using)




feature_states = [[], [], []]
for i in range(len(feature_states)):
	for j in range(6):
		feature_states[i].append([int((coords[1]-1) * j/5), int((coords[0]-1) * i/2)])
		#feature_states[i].append([int((coords[0]-1) * (1/6 + i/3)), int((coords[1]-1) * j/5)])
print feature_states
grids = grid(coords[0], coords[1], A, feature_states, probability= 1.0, sink = False)

for i in range(grids.y_max):
	for j in range(grids.x_max):
		index = 0
		for l in range(0, len(feature_states)):
			for m in range(0, len(feature_states[l])):
				grids.features[i, j, index + m] = math.exp(-0.5 * math.sqrt((i - feature_states[l][m][0])**2 + (j - feature_states[l][m][1])**2))
			index  = index + len(feature_states[l])

unsafe = []
for y in range(0, 6):
	for x in range(0, 1):
		unsafe.append([y, x])	
safety = 0.5

agent = car(states=grids.states)


start_s = observationToState(o, threshes)
start = grids.states[int(start_s/coords[0])][int(start_s%coords[0])]

grids.transitions = np.zeros([grids.y_max, grids.x_max, model.getA(), grids.y_max, grids.x_max]).astype(float)
for s in range(model.getS()):
	for a in range(model.getA()):
		p_sum = 0.0
		for s_ in range(model.getS()):
			p = model.getTransitionProbability(s, a, s_)
			if p > 0.0:
				grids.transitions[int(s/coords[0]), int(s%coords[0]), a, int(s_/coords[0]), int(s_%coords[0])] = p
				p_sum += p 

real=raw_input("Try Apprenticeship Learning? [Y/N]")
if real == 'Y' or real == 'y':
	print 'Yeah!!!'
	real=raw_input("Human demonstrate? [Y/N]")
	demo_mu = np.zeros(len(grids.features[-1][-1]))
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
			else:
				a = 1
			for i in range(combo):
				env.step(a)
	       	 	o1, rew, done, info = env.step(a);
			print o1
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


	starts = [np.array([0, 0])]
	cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe, toolbox)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': [start], 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': True, 'safety': None, 'unsafe': unsafe, 'toolbox': False, 'test_expert_train': update.expert_train}, {})
#	cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': [start], 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': False, 'safety': None, 'unsafe': unsafe, 'toolbox': False, 'test_expert_train': update.expert_train}, {})
#	cProfile.runctx('test_expert_train(grids, demo_mu, agent, starts, epsilon, steps, iteration, gamma, start_theta, MC, safety, unsafe)', {'grids': grids, 'demo_mu': demo_mu, 'agent': agent, 'epsilon': 1e-5, 'starts': [start], 'steps': steps, 'iteration': 30, 'gamma': gamma, 'start_theta': None, 'MC': False, 'safety': None, 'unsafe': unsafe, 'toolbox': True, 'test_expert_train': update.expert_train}, {})
	_, theta, policy ,_ = update.expert_train(grids, demo_mu, agent, epsilon = 1e-5, starts = [start], steps = steps, iteration = 30, gamma= gamma, start_theta = None, MC = True, safety = None, unsafe = unsafe)
	grids.w_features(theta)
	print grids.rewards
        print "where is the fxxx rewards?????"	
	o = env.reset()
	for t in xrange(steps):
		s =  observationToState(o, threshes); 
     		   # Convert the observation into our own space
      	 	 # Select the best action according to the policy
     		a = int(policy[int(s/coords[0]), int(s%coords[0])])
       		# Act
		for i in range(combo):
			env.step(a)
      	  	o1, rew, done, info = env.step(a);
		print o1
      	        # See where we arrived
     		s1 = observationToState(o1, threshes);

     	   	# Record information, and then run PrioritizedSweeping
        	exp.record(s, a, s1, rew);
        	model.sync(s, a, s1);
        	solver.stepUpdateQ(s, a);
        	solver.batchUpdateQ();

        	o = o1;

    		env.render()
	if record:
    		env.monitor.close()

for j in range(5):
	grids.w_features(theta)
	policy, _ = update.optimal_value(grids, agent, steps = steps, epsilon=1e-5, gamma = gamma)
	print policy
	o = env.reset()
	for t in xrange(steps):
		s =  observationToState(o, threshes); 
     		   # Convert the observation into our own space
      	 	 # Select the best action according to the policy
     		a = int(policy[int(s/coords[0]), int(s%coords[0])])
       		# Act
		for i in range(combo):
			env.step(a)
      	  	o1, rew, done, info = env.step(a);
		print o1
      	        # See where we arrived
     		s1 = observationToState(o1, threshes);

     	   	# Record information, and then run PrioritizedSweeping
        	exp.record(s, a, s1, rew);
        	model.sync(s, a, s1);
        	solver.stepUpdateQ(s, a);
        	solver.batchUpdateQ();

        	o = o1;

    		env.render()
	if record:
    		env.monitor.close()


real=raw_input("Try synthesizing? [Y/N]")
while real == 'Y' or real == 'y':
	_, _, policy, _ = update.expert_synthesize(grids, demo_mu, agent, [start], steps = steps, epsilon = 1e-5, iteration = 30, gamma = gamma, start_theta = theta, MC = True, unsafe =unsafe, safety = safety)

	o = env.reset()
	for t in xrange(steps):
		s =  observationToState(o, threshes); 
     		   # Convert the observation into our own space
      	 	 # Select the best action according to the policy
     		a = int(policy[int(s/coords[0]), int(s%coords[0])])
       		# Act
		for i in range(combo):
			env.step(a)
      	  	o1, rew, done, info = env.step(a);
		print o1
      	        # See where we arrived
     		s1 = observationToState(o1, threshes);

     	   	# Record information, and then run PrioritizedSweeping
        	exp.record(s, a, s1, rew);
        	model.sync(s, a, s1);
        	solver.stepUpdateQ(s, a);
        	solver.batchUpdateQ();

        	o = o1;

    		env.render()
	if record:
    		env.monitor.close()

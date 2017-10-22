import gym

# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.
import MDP

# Number of distretized pieces for each observation component
steps = 4;
# We disregard the cart position on the screen to make learning
# faster
threshes = [0.0, 0.4, 0.4, 0.4];
maxepisodes = 5000

# Gym parameters
render = 0;
record = 0;
recordfolder = './cartpole-experiment-4'

env = gym.make('CartPole-v0')

# Action space is 2, State space depends on steps
A = env.action_space.n;
S = env.observation_space.shape[0] * steps * steps * steps * steps;

# We are not going to assume anything here. We are just going to
# approximate the observation space in a finite number of states.
# In particular, we approximate each vector component in 4 steps.
# If we discard the first component (the cart position on the
# screen) we can learn faster, but adding it still works.
# Then we use PrioritizedSweeping in order to extract as much
# information as possible from each datapoint. Finally we select
# actions using a softmax policy.

exp = MDP.SparseExperience(S, A);
model = MDP.SparseRLModel(exp, 0.9);
solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
policy = MDP.QSoftmaxPolicy(solver.getQFunction(), 10.0 / 121);

def observationToState(o, thresh):
    s = int(0);
    exp = 1;
    for i in range(len(o)):
        if thresh[i] == 0:
            continue
        ox = (min(thresh[i], max(-thresh[i], o[i])) + thresh[i]);
        val = int((ox * float(steps)) / (2.0 * thresh[i]));
        if val == steps:
            val = steps - 1
        s += val * exp;
        exp *= steps;
    return s;

if record:
    env.monitor.start(recordfolder)

win = 0
# We use the following two variables to track how we are doing.
# Read more at the bottom
episodes=0
streak = list()
for i_episode in xrange(maxepisodes):
    o = env.reset()

    using = 0
    for t in xrange(200):
        if render or i_episode == maxepisodes - 1:
            env.render()

        # Convert the observation into our own space
        s = observationToState(o, threshes);
        # Select the best action according to the policy
        a = policy.sampleAction(s)
        # Act
        o1, rew, done, info = env.step(a);
        # See where we arrived
        s1 = observationToState(o1, threshes);

        if done:
            break;

        # Record information, and then run PrioritizedSweeping
        exp.record(s, a, s1, rew);
        model.sync(s, a, s1);
        solver.stepUpdateQ(s, a);
        solver.batchUpdateQ();

        o = o1;

    if render or i_episode == maxepisodes - 1:
        env.render()

    tag = '   ';
    # Here we have to set the reward since otherwise rewards are
    # always 1.0, so there would be no way for the agent to distinguish
    # between bad actions and good actions.
    if done:
        rew = -100;

    if t >= 199:
        tag = '###';
        win += 1;
        streak.append(1)
    else:
        streak.append(0)
    if len(streak) > 20:
        streak.pop(0)

    episodes +=1;
    exp.record(s, a, s1, rew);
    model.sync(s, a, s1);
    solver.stepUpdateQ(s, a);
    solver.batchUpdateQ();

    # If the learning process gets stuck in some local optima without
    # winning we just reset the learning. We don't want to try to change
    # what the agent has learned because this task is very easy to fail
    # when trying to learn something new (simple exploration will probably
    # just make the pole topple over). We just want to learn the correct
    # thing once and be done with it.
    if sum(streak) < 10 and episodes > 30:
        episodes = 0

        exp = MDP.SparseExperience(S, A);
        model = MDP.SparseRLModel(exp, 0.9);
        solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
        policy = MDP.QSoftmaxPolicy(solver.getQFunction(), 10.0 / 121);


    print "Episode {} finished after {} timesteps {} {} {}".format(i_episode, t+1, tag, win, using)

if record:
    env.monitor.close()

for s in range(model.getS()):
	print s, '\n'
	for a in range(model.getA()):
		print a, '\n'
		for s_ in range(model.getS()):
			p = model.getTransitionProbability(s, a, s_)
			if p > 0.0:
				print '(', s, ',', a, ',', s_, '): ', p, ' '
		print '\n'	
    	

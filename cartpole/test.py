import gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()
episode = 200
for i in range(episode):
	a = raw_input('Next step')
	a = int(a)
	if a > 1:
		a = 2
	elif a < 1:
		a = 0
	else:
		a = 1 
	info = env.step(a)
	print info
	env.render()
	

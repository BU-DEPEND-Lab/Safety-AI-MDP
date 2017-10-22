import gym
import re
import numpy as np

file = open("grid_world.dtmc", "r")
lines_str = file.readlines()
lines = []
initial = -1
from_ = 0
for i in range(len(lines_str)):
	line = []
	line_ = re.split(' |\n', lines_str[i])
	if line_[0] == str(from_) and initial == -1:
		initial = int(i)
	if i >= initial and initial != -1:
		for j in line_:
			if j != '':
				line.append(float(j))
		lines.append(line)
same_from_ = []
prob = 0.0
for i in range(0, len(lines)):
	if lines[i][0] == from_:
		prob += lines[i][-1]
	else:
		if prob > 1.0:
			print "inital ", from_, "sum of prob, ", prob, " > 1.0"	
		elif prob < 1.0:
			print "inital ", from_, "sum of prob, ", prob, " < 1.0"	
		from_ = lines[i][0]
		prob = lines[i][-1]

if prob != 1.0:
	print "inital ", from_, "sum of prob, ", prob, " != 1.0"	

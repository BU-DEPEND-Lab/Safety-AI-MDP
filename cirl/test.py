import subprocess, shlex
from threading import Timer
import os

def run(cmd, timeout_sec):
  kill_proc = lambda p: os.system('kill -9 $(pidof comics)')
  proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  timer = Timer(timeout_sec, kill_proc, [proc])
  try:
    timer.start()
    stdout, stderr = proc.communicate()
  finally:
    timer.cancel()
    print stdout
    print stderr
    prob = float("".join(stdout).split('\n')[-3].split(':')[-2].split(";")[0])
    print prob

def run1(cmd = ['sh', '/home/zekunzhou/workspace/prism-4.4.beta-linux64/bin/prism', './grid_world.pm', './grid_world.pctl'], timeout_sec = 5.0):
	kill_proc = lambda p: p.kill()
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
  	try:
    		timer.start()
   	 	stdout, stderr = proc.communicate()
  	finally:
    		timer.cancel()
		print stdout
		print stderr
  		try:
    			lines = "".join(stdout).split('\n')
    			for line in lines:
				print line.split(':')
				if line.split(':')[0] == 'Result':
					print line
					prob = float(line.split(':')[1].split('(')[0])
					break
			
			print float("".join(stdout).split('\n')[-7].split(' ')[1])
			if prob <= 1.0:
  				return prob
		except:
			return None
	
run1(['/home/zekunzhou/workspace/prism-4.4.beta-linux64/bin/prism', './grid_world.pm', './grid_world.pctl'], 5.0)
#run(['sh', '/home/zekunzhou/workspace/comics-1.0/comics.sh', './grid_world.conf', '--only_model_check'], 5.0)
'''
safety = 0.1
os.system('rm counter_example.path')
while safety > 0.0001:
	file = open('grid_world.conf', 'w')
	file.write('TASK counterexample\n')
	file.write('PROBABILITY_BOUND ' + str(safety) + '\n')
	file.write('DTMC_FILE grid_world.dtmc' + '\n')
	file.write('REPRESENTATION pathset' + '\n')
	file.write('SEARCH_ALGORITHM global' + '\n')
	file.write('ABSTRACTION concrete' + '\n')
	file.close()
	run(['sh', '/home/zekunzhou/workspace/comics-1.0/comics.sh', './grid_world.conf'], 5.0)
	try:
		file = open('counter_example.path', 'r')
		print "Generated counterexample of ", safety
		break
	except:
		print "No counterexample found for spec = ", safety, "shrinking down the safey"
		safety = safety / 10.0
'''

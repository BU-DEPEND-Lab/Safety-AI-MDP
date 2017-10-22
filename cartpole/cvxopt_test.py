from cvxopt import matrix, solvers
import numpy as np
c = matrix([2., 1., 1.])
G = [matrix([[ 0., 1., 0., 0.], [ 0., 0., 1., 0.], [0., 0., 0., 1.]])]
h = [matrix([1., 0., 0., 0.])]
#dims = {'l': 0, 'q': [4], 's': []}
sol = solvers.socp(c, Gq=G, hq=h)
sol['status']
print "hehe"
solution=sol['x']
	
print solution

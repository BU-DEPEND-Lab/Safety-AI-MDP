from cvxopt import matrix, solvers
import numpy as np
c = matrix([1.0])
G = matrix([ 1., -1.])
h = matrix([0.,  -1.])
dims = {'l': 2, 'q': [], 's': []}
sol = solvers.conelp(c, G, h, dims)
sol['status']
solution=sol['x']
solutions=[]
if solution==None:
	solutions.append(None)
print solutions
	

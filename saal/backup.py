
def expert_synthesize2(grids, expert, agent, starts, steps, epsilon=1e-6, iteration=100, gamma=0.5, start_theta= None, MC = False, unsafe = None, safety = 0.0001):
	print "Human demo feature ", expert
	print "Initial theta ", start_theta
	flag = None
	index = []
	mu_Bs = []
	mus = []
	policies = []	
	grids.w_features(start_theta)
	start_policy, start_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
	policies = policies + [start_policy]
	start_mu, MU = optimal_feature(grids, starts, steps, start_policy, epsilon = epsilon, gamma=gamma)
	#mus.append(start_mu)
	new_theta = np.array(start_theta)
	print "Model check ", start_theta
	p_B_sum = output_model(grids, starts, start_policy, steps, unsafe, safety)
	#new_mu = start_mu
	#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, start_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	print "Initial unsafe path rate ", p_B_sum

	print "model output finished for initial policy"
	if p_B_sum <= safety and p_B_sum is not None:
		return start_theta, start_policy, start_value, start_mu
	elif p_B_sum is None:
		print "Model checking failed"
		return None
	while p_B_sum > safety:
		new_mu_B =  counterexample(grids, MU, gamma, safety, epsilon)
		#new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
		if new_mu_B is not None:
			mu_Bs = mu_Bs + [new_mu_B]
			print "Add counterexample features ", new_mu_B	

		new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
		#new_index, new_theta, w_delta_mu = expert_update_theta(grids, new_mu, agent, steps, policies, mu_Bs, gamma, epsilon)
		grids.w_features(new_theta)
		print "Weight learnt from counterexamples ", new_theta

		new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
		print new_policy
		#print new_policy
		new_mu, MU = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
		print "Corresponding feature ", new_mu
		print "Model check ", new_theta
		p_B_sum = output_model(grids, starts, new_policy, steps, unsafe, safety)
		#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
		#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
		print "Policy unsafe rate ", p_B_sum
		print "Keep generating counterexamples until find a safe candidate" 
     	print "Found 1st safe policy towards safety ", new_theta	
	mus = mus + [new_mu]

	K = 1.0
	KK = 0.0
	k = 0.0
	mu = [[k,  1 - k],[mu_Bs, [expert]]]

		
	new_index, new_theta, w_delta_mu = expert_update_theta_v2(grids, mu, agent, steps, policies, mus, gamma, epsilon)
	#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
	grids.w_features(new_theta)
	print "Weight learnt from 1st combined feature ", new_theta
	
	new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)

	new_mu, MU = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
	print "Corresponding feature ", new_mu
	#mus = mus + [new_mu]

	#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
	#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
	#print "Policy unsafe rate ", p_B_sum
	print "Model check ", new_theta
	p_B_sum = output_model(grids, starts, new_policy, steps, unsafe, safety)

	i = 1	
	while True:
		print ">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", new_theta, "\nunsafe probability:", p_B_sum, "\nfeature ", new_mu, " = ", k, "*safe + ", 1 - k, "*expert\n" 
		file = open('log', 'a')
		file.write(">>>>>>>>> " + str(i) + "th iteration, candidate theta: "+ str(new_theta) + "; unsafe probability: " + str(p_B_sum) + "; feature " + str(new_mu) + " = " + str(k) + "*safe + " + str(1 - k) + "*expert\n") 
		file.close()
		#file = open('log', 'a')
		#file.write(">>>>>>>>> ", i, "th iteration\n", "candidate theta: ", new_theta, "\nunsafe prob:", p_B_sum, "\nfeature ", new_mu, " = ", k, "*safe + ", 1 - k, "*expert\n") 
		#file.close()
		i = i + 1
		
		if flag is not None and abs(K - k) < epsilon:
			print "Difference with the best one is too small", np.linalg.norm(flag['feature'] - new_mu, ord=2)
			#print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
			#print "Expert can get value ", np.dot(theta, expertw)
			#flag = [new_theta, new_policy, new_value, new_mu]
			#return grids, flag[0], flag[1], flag[2]
			break
		if p_B_sum is not None and p_B_sum > safety:
			print "Unsafe, learning from counterexample"
			#while p_B_sum > safety:
			#new_mu_B = np.sum(np.reshape(exp_u_B, [grids.y_max*grids.x_max, len(grids.features[-1][-1])]), 0) 
			new_mu_B =  counterexample(grids, MU, gamma, safety, epsilon)
			if new_mu_B is not None:
				mu_Bs = mu_Bs + [new_mu_B]
				print "Add counterexample features ", new_mu_B	

			#print "Keep generating counterexamples until find a safe candidate" 
			#new_index, new_theta, w_delta_mu = expert_update_theta(grids, np.zeros(len(grids.features[-1][-1])), agent, steps, policies, mu_Bs, gamma, epsilon)
			#grids.w_features(new_theta)
			#print "Weight learnt from counterexamples ", new_theta

			#new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			#print "Policy unsafe rate ", p_B_sum
			#print "Found safe weight towards safety ", new_theta
			#new_mu = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			#mus = mus + [new_mu]
			KK = k
			k = (K + KK)/2			
			mu = [[k,  1 - k],[mu_Bs, [expert]]]
			#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			new_index, new_theta, w_delta_mu = expert_update_theta_v2(grids, mu, agent, steps, policies, mus, gamma, epsilon)
			grids.w_features(new_theta)
			print "Weight learnt from combined feature ", new_theta

			new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)

			new_mu, MU = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", new_mu
			#mus = mus + [new_mu]
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Model check ", new_theta
			p_B_sum = output_model(grids, starts, new_policy, steps, unsafe, safety)
			print "Policy unsafe rate ", p_B_sum
			#mus = mus + [new_mu]
			while p_B_sum > safety and np.linalg.norm(new_mu_B - mu_Bs[-1], ord = 2) > epsilon:
				new_mu_B =  counterexample(grids, MU, gamma, safety, epsilon)
				print "Unsafe, learning from counterexample"
		
				mu = [[k,  1 - k],[mu_Bs, [expert]]]
				new_index, new_theta, w_delta_mu = expert_update_theta_v2(grids, mu, agent, steps, policies, mus, gamma, epsilon)
				grids.w_features(new_theta)
				print "Weight learnt from combined feature ", new_theta

				new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)

				new_mu, MU = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
				print "Corresponding feature ", new_mu
			
				print "Model check ", new_theta
				p_B_sum = output_model(grids, starts, new_policy, steps, unsafe, safety)
				print "Policy unsafe rate ", p_B_sum
		elif p_B_sum is not None:	
			print "Safe, learning from expert"
			mus = mus + [new_mu]
			#if flag is not None and np.linalg.norm(expert - new_mu, ord=2)  < np.linalg.norm(expert - flag['feature'], ord=2):
			if flag is not None and np.dot(new_theta, expert) > np.dot(flag['weight'], expert):
				flag = {'weight': new_theta, 'policy': new_policy, 'value': new_value, 'unsafe': p_B_sum, 'feature': new_mu}
				print "New best candidate"
				#print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(new_theta, expert)
				K = k
				k = (K + KK)/2
			elif flag is None:
				flag = {'weight': new_theta, 'policy': new_policy, 'value': new_value, 'unsafe': p_B_sum, 'feature': new_mu}
				print "1st best candidate"
				print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(theta, expert)
				K = k
				k = (K + KK)/2

			else:
				print "Not the best"
				print "Feature deviation from expert ", np.linalg.norm(new_mu - expert, ord = 2)
				#print "Expert can get value ", np.dot(theta, expert)
				k = (K + k)/2

			print "Add new candidate policy expected feature", new_mu
			
			mu = [[k,  1 - k],[mu_Bs, [expert]]]
			#grids, new_theta, new_policy, new_value = expert_train_v1(grids, mu, agent, starts, steps, epsilon=epsilon, iteration=30, gamma=gamma, start_theta= new_theta, MC = False, safety = None)
			new_index, new_theta, w_delta_mu = expert_update_theta_v2(grids, mu, agent, steps, policies, mus, gamma, epsilon)
			grids.w_features(new_theta)
			print "Weight learnt from combined feature ", new_theta
			new_policy, new_value = optimal_value(grids, steps = steps, epsilon=epsilon, gamma = gamma)
			policies = policies + [new_policy]

			new_mu, MU = optimal_feature(grids, starts, steps, new_policy, epsilon = epsilon, gamma=gamma)
			print "Corresponding feature ", new_mu
			#mus = mus + [new_mu]
			
			print "Model check ", new_theta
			p_B_sum = output_model(grids, starts, new_policy, steps, unsafe, safety)
			#exp_u_G, p_G, exp_u_B, p_B, _ = sample_feature(grids, agent, starts, grids.x_max*grids.y_max, new_policy, epochs= 5000, epsilon = 1e-3, gamma=gamma)
			#p_B_sum = np.sum(np.reshape(p_B, [grids.y_max*grids.x_max]))/(len(starts))
			print "Policy unsafe rate ", p_B_sum
		else:
			print "Model checking failed"
			return None

	print "Iteration ended, best safe theta ", flag['weight']
	print "It's unsafe probability is ", flag['unsafe']
	print "Distance to expert feature ", np.linalg.norm(expert - flag['feature'], ord= 2)
	return grids, flag['weight'], flag['policy'], flag['value']

def train(sess, env, args, actors, critics, noise):
	# global variables initializer
	# 
	init = tf.global_variables_initializer()
	sess.run(init)

	# First, update target network
	for actor in actors:
		actor.update_target()
	for critic in critics:
		critic.update_target()

	# Initialize replay memory
	replayMemory = ReplayMemory(int(args["buffer_size"], args["random_seed"]))

	# Start training for each episode
	for ep in range(int(args["max_episodes"])):
		# Reset environment
		s = env.reset()
		# One episode matrix(agent number) for saving reward of each agent 
		episode_reward = np.zeros((env.n, ))

		# for each step
		for step in range(int(args["max_steps"])):
			# rendering
			env.render()
		
		# store actions
		actions = []

		# for each agent to get actions
		for i in range(env.n):
			# Actor predicts action using observation with noise
			actions.append(actors[i].act(np.reshape(s[i], (-1, actor[i].state_dim)), noise[i]()))

		# get next_state, reward, doneFlag, xx from env.step, using actions
		next_s, reward, done, _ = env.step(actions)

		# Add (state, actions, reward, done, next_state) to replay memory
		replayMemory.add(s, actions, reward, done, next_s)

		# current state change to next state
		s = next_s

		# For each agent, XXX
		for i in range(env.n):
			actor = actors[i]
			critic = critics[i]
			
			# Size is greater than batch size
			if replayMemory.size() > int(args["minibatch_size"]):

				s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))

				# actions 
				a = []

				# for each actor, target network (batch_state) to predict action
				for j in range(env.n):
					# each agent batch state
					state_batch_j = np.asarray([x for x in s_batch[ :,j]])
					a.append(actors[j].predict_target(state_batch_j))

				# temporary actions
				a_temp = np.asarray([x.flatten() for x in a_temp])

				# action for critic to evaluate
				a_for_critic = np.asarray([x.flatten() for x in a_temp])

				# 
				s2_batch_i = np.asarray([x for x in s2_batch[:, i]])

				# 
				targetQ = critic.predict_target(s2_batch_i, a_for_critic)

				yi = []

				for k in range(int(args["minibatch_size"])):
					# if game over
					if d_batch[:, i][k]:
						yi.append(r_batch[:, i][k])
					# otherwise, current Q plus future reward * discounted factor
					else:
						yi.append(r_batch[:, i][k] + critic.gamma * targetQ[k])

				s_batch_i = np.asarray([x for x in s_batch[:, i]])

				critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch]), np.asarray(yi))

				








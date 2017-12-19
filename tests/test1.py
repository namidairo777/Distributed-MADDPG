import queue

def func1():
	print("func1")
	
	print(get())
x = None



"""
		Collect workers' buffer (batch data) using queue in multi-threading programming
		Using mini batch (s, a, r, d, s2) to update critic and actor

		1. Update old policy
		2. Calculate advantage 
		3. Actor update
		4. Critic update

		The question here is that what kind of advantage can make
		attribution to our multi-agent cooperation.
		In DPPO, the advantage is the loss between yi and critic value

		v = dense 100 (state)
		dc_r = 
		advantage = dc_r - v

		CriticLoss = advantage
		Critic update Adam optimizer(CriticLoss)


		ratio = pi(a, s) / oldpi(a, s)
		Clipped surrogate objective
		Loss = min(ratio * adv, clip(ratio, 1-0.2, 1+0.2) * advantage)
		Actor optimization Adam optimizer(-loss)
		
		###########
		##  Actor 
		###########
		Input = state_dim
		Dense(400) -> Relu
		Dense(300) -> Relu
		sigma = Dense(action_dim)
		mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
		tf.distributions.Normal(loc=mu, scale=sigma)


		###########
		##  Critic
		##########
		Input1 = state_dim
		Input2 = action_dim
		obs = Dense(400)(input1) -> relu
		actions = Dense(300)(input2) -> relu
		temp_obs = Dense(300)(obs)
		Add()(temp_obs, actions) -> relu


		actor loss = 


		"""
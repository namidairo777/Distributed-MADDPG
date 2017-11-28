import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Concatenate, Activation
from keras.optimizers import Adam
import keras.backend as K



class Brain(object):
	def __init__(self):
		pass

	def update(self):
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
		"""




###########################
#####    WORKER    ########
###########################

class Worker(object):
	# init
	def __init__(self, wid, max_episodes, max_episode_len):
		self.wid = wid
		self.env = ma.make_env("simple_tag")
		self.brain = BRAIN
		self.max_episodes = max_episodes
		self.max_episode_len = max_episode_len

	def work(self):
		global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
		while not COORD.should_stop():
			s = self.env.reset()
			episode_reward = 0
			buffer_s, buffer_a, buffer_r = [], [], []

			for stp in range(self.max_episode_len):
				if not ROLLING_EVENT.is_set():
					ROLLING_EVENT.wait()
					buffer_s, buffer_a, buffer_r = [], [], []

				actions = []
				for i in range(self.brain.actors):
					actions.append(self.brain.actors[i].predict(s[i])) 

				s, r, done, s2 = self.env.step(actions)

				buffer_s.append(s)
				buffer_a.append(actions)
				buffer_r.append(r)

				s = s2
				episode_reward += r

				GLOBAL_UPDATE_COUNTER += 1

				if stp == self.max_episode_len - 1 or GLOBAL_UPDATE_COUNTER > BATCH_SIZE:
					"""
					Get value from brain
					Get value from brain
					get Value from barin
					Calculate dic_v in time order using brain.getValue()
					discounted_value = r + gamma * v_s
					bs, ba, br = buffer_s, buffer_a, discounted_value
					queue.put(np.hstack(bs, ba, br))
					"""
					if GLOBAL_UPDATE_COUNTER >= BATCH_SIZE:
						ROLLING_EVENT.clear()
						UPDATE_EVENT.is_set()

					if GLOBAL_EP > self.max_episode_len:
						COORD.request_stop():
						break

			GLOBAL_EP += 1


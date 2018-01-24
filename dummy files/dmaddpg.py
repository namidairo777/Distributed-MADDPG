import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Concatenate, Activation
from keras.optimizers import Adam
import keras.backend as K



class Brain(object):
	def __init__(self, actors, critics, controller):
		self.actors = actors
		self.critics = critics
		self.controller = controller

	def update(self):
		"""
		
		"""


		s_batch, a_batch, r_batch, d_batch, s2_batch = getFromQueue(GLOBAL_QUEUE)
		action_dims_done = 0
		for i in range(ave_n):
			Actor = actors[i]
			critic = critics[i]
			if replayMemory.size()>int(args['minibatch_size']):
				s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
				a = []
				for j in range(ave_n):
					state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
					a.append(actors[j].predict_target(state_batch_j))
				#print(np.asarray(a).shape)
				a_temp = np.transpose(np.asarray(a),(1,0,2))
				#print("a_for_critic", a_temp.shape)
				a_for_critic = np.asarray([x.flatten() for x in a_temp])
				s2_batch_i = np.asarray([x for x in s2_batch[:,i]]) # Checked till this point, should be fine.
				# print("s2_batch_i", s2_batch_i.shape)
				targetQ = critic.predict_target(s2_batch_i,a_for_critic) # Should  work, probably
				yi = []
				for k in range(int(args['minibatch_size'])):
					if d_batch[:,i][k]:
						yi.append(r_batch[:,i][k])
					else:
						yi.append(r_batch[:,i][k] + critic.gamma*targetQ[k])
				s_batch_i = np.asarray([x for x in s_batch[:,i]])
				
				# critic.train()
				#critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch]),np.asarray(yi))
				loss = critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch[:, 0: ave_n, :]]),np.asarray(yi))
				losses.append(loss)
				# callback.set_model(critic.mainModel)
				# write_log(callback, train_names, logs, ep)
				#predictedQValue = critic.train(s_batch,np.asarray([x.flatten() for x in a_batch]),yi)
				#episode_av_max_q += np.amax(predictedQValue)
				
				actions_pred = []
				# for j in range(ave_n):
				for j in range(ave_n):
					state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
					actions_pred.append(actors[j].predict(state_batch_j)) # Should work till here, roughly, probably
				a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
				a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
				s_batch_i = np.asarray([x for x in s_batch[:,i]])
				grads = critic.action_gradients(s_batch_i,a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
				actor.train(s_batch_i,grads)
				#print("Training agent {}".format(i))
				actor.update_target()
				critic.update_target()
			action_dims_done = action_dims_done + actor.action_dim
		
		# Only DDPG agent
		for i in range(ave_n, env.n):
			actor = actors[i]
			critic = critics[i]
			if replayMemory.size() > int(args["minibatch_size"]):
				s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))
				
				# action for critic					
				s_batch_i = np.asarray([x for x in s_batch[:,i]])
				action = np.asarray(actor.predict_target(s_batch_i))
				#print("action", action.shape)
				# a_temp = np.transpose(np.asarray(a),(1,0,2))
				# a_for_critic = np.asarray([x.flatten() for x in a_temp])
				# for j in range(env.n):
				#    print(np.asarray([x for x in s_batch[:,j]]).shape)
				action_for_critic = np.asarray([x.flatten() for x in action])
				s2_batch_i = np.asarray([x for x in s2_batch[:, i]])
				# critic.predict_target(next state batch, actor_target(next state batch))
				targetQ = critic.predict_target(s2_batch_i, action_for_critic)
				#print("length: ", len(targetQ))
				#print(targetQ)
				
				#time.sleep(10)
				# loss = meanSquare(y - Critic(batch state, batch action)
				# y = batch_r + gamma * targetQ
				y_i = []
				for k in range(int(args['minibatch_size'])):
					# If ep is end
					if d_batch[:, i][k]:
						y_i.append(r_batch[:, i][k])
					else:
						y_i.append(r_batch[:, i][k] + critic.gamma * targetQ[k])
				# state batch for agent i
				s_batch_i= np.asarray([x for x in s_batch[:, i]])
				loss = critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))
				losses.append(loss)
				# callback.set_model(critic.mainModel)
				# write_log(callback, train_names, logs, ep)
				action_for_critic_pred = actor.predict(s2_batch_i)
				gradients = critic.action_gradients(s_batch_i, action_for_critic_pred)[:, :]
				# check gradients
				"""
				grad_check = tf.check_numerics(gradients, "something wrong with gradients")
				with tf.control_dependencies([grad_check]):
 						
					actor.train(s_batch_i, gradients)
				"""
				actor.train(s_batch_i, gradients)
				
				actor.update_target()
				critic.update_target()

###########################
#####    WORKER    ########
###########################

class Worker(object):
	# init
	def __init__(self, wid, brain, controller,  max_episode_len, batch_size, seed):
		self.wid = wid
		self.env = ma.make_env("simple_tag")
		self.env.seed(int(seed))
		self.brain = brain
		self.controller = controller
		self.max_episode_len = max_episode_len
		self.batch_size = batch_size

	def work(self):
		# global GLOBAL_EP, GLOBAL_UPDATE_COUNTER
		while not COORD.should_stop():
			s = self.env.reset()
			episode_reward = 0
			for stp in range(args['max_episode_len']):
				if not ROLLING_EVENT.is_set():
					ROLLING_EVENT.wait()

				actions = []
				for i in range(self.brain.actors):
					actor = self.barin.actors[i]
					actions.append(actor.act(state_input, noise[i]()).reshape(actor.action_dim,)) 

				s, r, done, s2 = self.env.step(actions)

				Q.put([s, actions, r, d, s2])

				s = s2
				episode_reward += r

				GLOBAL_UPDATE_COUNTER += 1

				if stp == self.max_episode_len - 1 or GLOBAL_UPDATE_COUNTER >= self.batch_size:
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


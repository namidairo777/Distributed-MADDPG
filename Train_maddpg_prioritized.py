import numpy as np
import gym
import random
from ReplayMemory import ReplayMemory
from keras.callbacks import TensorBoard
import time, os
import tensorflow as tf
#from actorcritic import ActorNetwork,CriticNetwork

def build_summaries(n):
	#episode_reward = tf.get_variable("episode_reward",[1,n])
	# record reward summay 
	# ave_reward = tf.Variable(0.)
	# good_reward = tf.Variable(0.)
	# episode_reward =   tf.Variable(0.)
	# tf.summary.scalar("Ave_Reward",ave_reward)
	# tf.summary.scalar("Good_Reward",good_reward)

	rewards = [tf.Variable(0.) for i in range(n)]

	for i in range(n):
		tf.summary.scalar("Reward_Agent" + str(i), rewards[i])
	
	#episode_ave_max_q = tf.Variable("episode_av_max_")
	#tf.summary.scalar("QMaxValue",episode_ave_max_q)
	#summary_vars = [episode_reward,episode_ave_max_q]
	# summary_vars = [ave_reward, good_reward]
	summary_vars = rewards
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def train(sess,env,args,actors,critics,noise, ave_n):

	summary_ops,summary_vars = build_summaries(env.n)
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
	
	# callbacks = []
	# train_names = ['train_loss', 'train_mae']
	# callback = TensorBoard(args['summary_dir'])

	for actor in actors:
		actor.update_target()
	for critic in critics:
		critic.update_target()
	
	#for i in range(20):
	#		print([noise[i]()for i in range(env.n)])


	
	replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

	for ep in range(int(args['max_episodes'])):

		start = time.time()

		s = env.reset()
		episode_reward = np.zeros((env.n,))
		#episode_av_max_q = 0

		for stp in range(int(args['max_episode_len'])):
			
			action_dims_done = 0

			if args['render_env']:
				env.render()
			
			a = []

			for i in range(env.n):
				actor = actors[i]
				state_input = np.reshape(s[i],(-1,actor.state_dim))
				a.append(actor.act(state_input, noise[i]()).reshape(actor.action_dim,))
			# print(a)
			#time.sleep(10)
			s2,r,done,_ = env.step(a) # a is a list with each element being an array
			#replayMemory.add(np.reshape(s,(actor.input_dim,)),np.reshape(a,(actor.output_dim,)),r,done,np.reshape(s2,(actor.input_dim,)))
			if ep % 50 == 0:
				env.render()
			replayMemory.add(s,a,r,done,s2)			
			s = s2
			# MADDPG Adversary Agent			
			for i in range(ave_n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size() > int(args['m_size']):
					s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args['m_size']))
					a = []
					for j in range(ave_n):
						state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
						a.append(actors[j].predict_target(state_batch_j))
					a_temp = np.transpose(np.asarray(a),(1,0,2))
					a_for_critic = np.asarray([x.flatten() for x in a_temp])
					s2_batch_i = np.asarray([x for x in s2_batch[:,i]]) 
					targetQ = critic.predict_target(s2_batch_i,a_for_critic)
					yi = []
					for k in range(int(args['m_size'])):
						if d_batch[:,i][k]:
							yi.append(r_batch[:,i][k])
						else:
							yi.append(r_batch[:,i][k] + critic.gamma*targetQ[k])
					# a2 = actor.predict_target(s_batch)
					# Q_target = critic.predict_target(s2_batch, a2)
					# y = r + gamma * Q_target
					# TD loss = yi - critic.predict(s_batch, a_batch)				
					s_batch_i = np.asarray([x for x in s_batch[:,i]])
					a_batch_data = np.asarray([x.flatten() for x in a_batch[:, 0: ave_n, :]])
					target_q = np.asarray(yi)
					# loss = batch
					losses = []
					# clip
					index = 0
					# number of losses
					loss_num = int(int(args['m_size']) / int(args['n_size']))
					for i in range(loss_num):
						loss = critic.get_loss(s_batch_i[index:index+int(args["n_size"])], 
											   a_batch_data[index:index+int(args["n_size"])], 
											   target_q[index:index+int(args["n_size"])])
						losses.append(loss)
						index += int(args["n_size"])
					# which has max loss
					sorted_index = np.argsort(losses).tolist()
					max_index = sorted_index[-1]
					# clip index
					head = max_index * int(args["n_size"])
					tail = head + int(args["n_size"])
					# clipped batch data with higher losses
					prioritized_a_batch = a_batch_data[head: tail] 
					prioritized_s_batch = s_batch_i[head: tail] 
					prioritized_target_q = target_q[head: tail]
					# critic train
					critic.train(prioritized_s_batch, prioritized_a_batch, prioritized_target_q)
					actions_pred = []
					# for j in range(ave_n):
					for j in range(ave_n):
						state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
						actions_pred.append(actors[j].predict(state_batch_j[head: tail])) 
					a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
					a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
					grads = critic.action_gradients(prioritized_s_batch, a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
					# actor train
					actor.train(prioritized_s_batch, grads)
				action_dims_done = action_dims_done + actor.action_dim
			# Only DDPG agent			
			for i in range(ave_n, env.n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size() > int(args["minibatch_size"]):
					s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))									
					s_batch_i = np.asarray([x for x in s_batch[:,i]])
					action = np.asarray(actor.predict_target(s_batch_i))
					action_for_critic = np.asarray([x.flatten() for x in action])
					s2_batch_i = np.asarray([x for x in s2_batch[:, i]])
					targetQ = critic.predict_target(s2_batch_i, action_for_critic)
					y_i = []
					for k in range(int(args['minibatch_size'])):
						if d_batch[:, i][k]:
							y_i.append(r_batch[:, i][k])
						else:
							y_i.append(r_batch[:, i][k] + critic.gamma * targetQ[k])
					s_batch_i= np.asarray([x for x in s_batch[:, i]])
					critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))
					action_for_critic_pred = actor.predict(s2_batch_i)
					gradients = critic.action_gradients(s_batch_i, action_for_critic_pred)[:, :]
					actor.train(s_batch_i, gradients)			
			for i in range(0, env.n):
				actor = actors[i]
				critic = critics[i]
				actor.update_target()
				critic.update_target()
			
			episode_reward += r
			#print(done)
			if stp == int(args["max_episode_len"])-1 or np.all(done) :
				
				ave_reward = 0.0
				good_reward = 0.0
				for i in range(env.n):
					if i < ave_n:
						ave_reward += episode_reward[i]
					else:
						good_reward += episode_reward[i]
				
				#summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(stp)})
				summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: ave_reward, summary_vars[1]: good_reward})
				# summary_str = sess.run(summary_ops, feed_dict = {summary_vars[i]: losses[i] for i in range(len(losses))})
				writer.add_summary(summary_str, ep)
				writer.flush()
				# print ('|Reward: {:d}| Episode: {:d}| Qmax: {:.4f}'.format(int(episode_reward),ep,(episode_av_max_q/float(stp))))
				showReward(episode_reward, env.n, ep, start)
				break

			#if stp == int(args['max_episode_len'])-1:
				#showReward(episode_reward, env.n, ep)

		# save model
		if ep % 50 == 0 and ep != 0:
			print("Starting saving model weights every 50 episodes")
			for i in range(env.n):
				# saveModel(actors[i], i, args["modelFolder"])
				saveWeights(actors[i], i, args["modelFolder"])
			print("Model weights saved")

		if ep % 200 == 0 and ep != 0:
			directory = args["modelFolder"] + "ep" + str(ep) + "/"
			if not os.path.exists(directory):
				os.makedirs(directory)
			print("Starting saving model weights to folder every 200 episodes")
			for i in range(env.n):
				# saveModel(actors[i], i, args["modelFolder"])
				saveWeights(actors[i], i, directory)
			print("Model weights saved to folder")



def saveModel(actor, i, pathToSave):
	actor.mainModel.save(pathToSave + str(i) + ".h5")

def saveWeights(actor, i, pathToSave):
	actor.mainModel.save_weights(pathToSave + str(i) + "_weights.h5")

def showReward(episode_reward, n, ep, start):
	reward_string = ""
	for re in episode_reward:
		reward_string += " {:5.2f} ".format(re)
	print ('|Episode: {:4d} | Time: {:2d} | Rewards: {:s}'.format(ep, int(time.time() - start), reward_string))

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
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
		# callback = TensorBoard(args['summary_dir'])
		# callback.set_model(critic.mainModel)
		# callbacks.append(callback)

		critic.update_target()
	
	replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

	for ep in range(int(args['max_episodes'])):

		start = time.time()

		s = env.reset()
		episode_reward = np.zeros((env.n,))
		#episode_av_max_q = 0

		for stp in range(int(args['max_episode_len'])):

			losses = []
			# action_dims_done = 0

			if args['render_env']:
				env.render()
			
			a = []

			for i in range(env.n):
				actor = actors[i]
				state_input = np.reshape(s[i],(-1,actor.state_dim))
				a.append(actor.act(state_input, noise[i]()).reshape(actor.action_dim,))
						
			s2,r,done,_ = env.step(a) # a is a list with each element being an array
			#replayMemory.add(np.reshape(s,(actor.input_dim,)),np.reshape(a,(actor.output_dim,)),r,done,np.reshape(s2,(actor.input_dim,)))
			replayMemory.add(s,a,r,done,s2)
			s = s2

			
			# Only DDPG agent
			
			for i in range(env.n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size() > int(args["minibatch_size"]):
					s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))
					
					# action for critic					
					s_batch_i = np.asarray([x for x in s_batch[:,i]])

					action = np.asarray(actor.predict_target(s_batch_i))

					action_for_critic = np.asarray([x.flatten() for x in action])

					s2_batch_i = np.asarray([x for x in s2_batch[:, i]])

					# critic.predict_target(next state batch, actor_target(next state batch))
					targetQ = critic.predict_target(s2_batch_i, action_for_critic)

					y_i = []
					for k in range(int(args['minibatch_size'])):
						# If ep is end
						if d_batch[:, i][k]:
							y_i.append(r_batch[:, i][k])
						else:
							y_i.append(r_batch[:, i][k] + critic.gamma * targetQ[k])
					# state batch for agent i
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


		# print("Cost Time: ", int(time.time() - start), "s")


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
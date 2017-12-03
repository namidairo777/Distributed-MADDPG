import numpy as np
import gym
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
import time
#from actorcritic import ActorNetwork,CriticNetwork

def build_summaries(n):
	#episode_reward = tf.get_variable("episode_reward",[1,n])
	# record reward summay 
	ave_reward = tf.Variable(0.)
	good_reward = tf.Variable(0.)
	# episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Ave_Reward",ave_reward)
	tf.summary.scalar("Good_Reward",good_reward)

	
	#episode_ave_max_q = tf.Variable("episode_av_max_")
	#tf.summary.scalar("QMaxValue",episode_ave_max_q)
	#summary_vars = [episode_reward,episode_ave_max_q]
	summary_vars = [ave_reward, good_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def train(sess,env,args,actors,critics,noise, ave_n):

	summary_ops,summary_vars = build_summaries(env.n)
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter(args['summary_dir'],sess.graph)

	for actor in actors:
		actor.update_target()
	for critic in critics:
		critic.update_target()
	
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
						
			s2,r,done,_ = env.step(a) # a is a list with each element being an array
			#replayMemory.add(np.reshape(s,(actor.input_dim,)),np.reshape(a,(actor.output_dim,)),r,done,np.reshape(s2,(actor.input_dim,)))
			replayMemory.add(s,a,r,done,s2)
			s = s2

			# MADDPG Adversary Agent			
			for i in range(ave_n):

				actor = actors[i]
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
					critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch[:, 0: ave_n, :]]),np.asarray(yi))
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

					critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))

					

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
			

			
			episode_reward += r
			#print(done)
			if stp == int(args["max_episode_len"])-1 or np.all(done) :
				ave_reward = 0.0
				good_reward = 0.0
				for i in range(env.n):
					if i < ave_n - 1:
						ave_reward += episode_reward[i]
					else:
						good_reward += episode_reward[i]
				#summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(stp)})
				# summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: ave_reward, summary_vars[1]: good_reward})
				#writer.add_summary(summary_str,ep)
				

				#writer.flush()

				#print ('|Reward: {:d}| Episode: {:d}| Qmax: {:.4f}'.format(int(episode_reward),ep,(episode_av_max_q/float(stp))))
				showReward(episode_reward, env.n, ep)
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

		print("Cost Time: ", int(time.time() - start), "s")


def saveModel(actor, i, pathToSave):
	actor.mainModel.save(pathToSave + str(i) + ".h5")

def saveWeights(actor, i, pathToSave):
	actor.mainModel.save_weights(pathToSave + str(i) + "_weights.h5")

def showReward(episode_reward, n, ep):
	print ('|Episode: {:d}	| Rewards: {:5.2f} {:5.2f} {:5.2f}'.format(ep, episode_reward[0], episode_reward[1], episode_reward[2]))



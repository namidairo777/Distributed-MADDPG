import numpy as np
import gym
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
#from actorcritic import ActorNetwork, CriticNetwork

def build_summaries(n):
	#episode_reward = tf.get_variable("episode_reward", [1, n])
	episode_reward =   tf.Variable(0.)
	tf.summary.scalar("Reward", episode_reward)
	#episode_ave_max_q = tf.Variable("episode_av_max_")
	#tf.summary.scalar("QMaxValue", episode_ave_max_q)
	#summary_vars = [episode_reward, episode_ave_max_q]
	summary_vars = [episode_reward]
	summary_ops = tf.summary.merge_all()
	return summary_ops,  summary_vars

def train(sess, env, args, actors, critics, noise):

	summary_ops, summary_vars = build_summaries(env.n)
	init = tf.global_variables_initializer()
	sess.run(init)
	writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

	for actor in actors:
		actor.update_target()
	for critic in critics:
		critic.update_target()
	
	# initiate experience replay
	replayMemory = ReplayMemory(int(args['buffer_size']),  int(args['random_seed']))

	# for each episode
	for ep in range(int(args['max_episodes'])):

		s = env.reset()
		episode_reward = np.zeros((env.n,  ))
		#episode_av_max_q = 0

		for stp in range(int(args['max_episode_len'])):
			if args['render_env']:
				env.render()

			# actions with noise
			a = []
			action_dims_done = 0

			for i in range(env.n):
				actor = actors[i]
				a.append(actor.act(np.reshape(s[i],  (-1,  actor.state_dim)),  noise[i]()).reshape(actor.action_dim,  ))
						
			s2,  r,  done,  _ = env.step(a) # a is a list with each element being an array
			#replayMemory.add(np.reshape(s,  (actor.input_dim,  )),  np.reshape(a,  (actor.output_dim,  )),  r,  done,  np.reshape(s2,  (actor.input_dim,  )))
			replayMemory.add(s,  a,  r,  done,  s2)
			s = s2

			for i in range(env.n):
				actor = actors[i]
				critic = critics[i]
				if replayMemory.size() > int(args['minibatch_size']):

					s_batch,  a_batch,  r_batch,  d_batch,  s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
					# actors predict actions (without noise)
					a = []
					for j in range(env.n):
						state_batch_j = np.asarray([x for x in s_batch[:,  j]]) #batch processing will be much more efficient even though reshaping will have to be done
						a.append(actors[j].predict_target(state_batch_j))

					a_temp = np.transpose(np.asarray(a),  (1, 0, 2))
					a_for_critic = np.asarray([x.flatten() for x in a_temp])
					s2_batch_i = np.asarray([x for x in s2_batch[:, i]]) # Checked till this point,  should be fine.
					targetQ = critic.predict_target(s2_batch_i, a_for_critic) # Should  work,  probably

					yi = []
					for k in range(int(args['minibatch_size'])):
						if d_batch[:, i][k]:
							yi.append(r_batch[:, i][k])
						else:
							yi.append(r_batch[:, i][k] + critic.gamma*targetQ[k])
					s_batch_i = np.asarray([x for x in s_batch[:, i]])
					critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch]), np.asarray(yi))
					#predictedQValue = critic.train(s_batch, np.asarray([x.flatten() for x in a_batch]), yi)
					#episode_av_max_q += np.amax(predictedQValue)
					
					actions_pred = []
					for j in range(env.n):
						state_batch_j = np.asarray([x for x in  s2_batch[:, j]])
						actions_pred.append(actors[j].predict(state_batch_j)) # Should work till here,  roughly,  probably

					a_temp = np.transpose(np.asarray(actions_pred), (1, 0, 2))
					a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
					s_batch_i = np.asarray([x for x in s_batch[:, i]])
					grads = critic.action_gradients(s_batch_i, a_for_critic_pred)[:, action_dims_done:action_dims_done + actor.action_dim]
					actor.train(s_batch_i, grads)
					#print("Training agent {}".format(i))
					actor.update_target()
					critic.update_target()

			action_dims_done = action_dims_done + actor.action_dim
			episode_reward += r
			if np.all(done):
				#summary_str = sess.run(summary_ops,  feed_dict = {summary_vars[0]: episode_reward,  summary_vars[1]: episode_av_max_q/float(stp)})
				summary_str = sess.run(summary_ops,  feed_dict = {summary_vars[0]: np.sum(episode_reward)})
				writer.action_dims_donesummary(summary_str, ep)
				writer.flush()
				#print ('|Reward: {:d}| Episode: {:d}| Qmax: {:.4f}'.format(int(episode_reward), ep, (episode_av_max_q/float(stp))))
				print ('|Reward: {:d}, {:d}, {:d}, {:d}	| Episode: {:d}'.format(int(episode_reward[0]), int(episode_reward[1]), int(episode_reward[2]), int(episode_reward[3]), ep))
				break



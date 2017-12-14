import numpy as np
import gym, threading, queue
import make_env as ma
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
from keras.callbacks import TensorBoard
# from dmaddpg import Brain, Worker
import time, os
#from actorcritic import ActorNetwork,CriticNetwork

class Brain(object):
	def __init__(self, actors, critics, ave_n, env_n):
		self.actors = actors
		self.critics = critics
		self.ave_n = ave_n
		self.env_n = env_n

	def update(self):
		global global_step, coord, global_step_max, update_event, rolling_event
		while not coord.should_stop():
			if global_step < global_step_max: 
				update_event.wait()
				global global_queue
				s_batch, a_batch, r_batch, d_batch, s2_batch = [], [], [], [], []
				for i in range(global_queue.qsize()):
					data = global_queue.get()
					s_batch.append(data[0])
					a_batch.append(data[1])
					r_batch.append(data[2])
					d_batch.append(data[3])
					s2_batch.append(data[4])
				action_dims_done = 0
				for i in range(self.ave_n):
					Actor = self.actors[i]
					critic = self.critics[i]
					if True: #replayMemory.size()>int(args['minibatch_size']):
						# s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
						a = []
						for j in range(self.ave_n):
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
						loss = critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch[:, 0: self.ave_n, :]]),np.asarray(yi))
						losses.append(loss)
						# callback.set_model(critic.mainModel)
						# write_log(callback, train_names, logs, ep)
						#predictedQValue = critic.train(s_batch,np.asarray([x.flatten() for x in a_batch]),yi)
						#episode_av_max_q += np.amax(predictedQValue)
						
						actions_pred = []
						# for j in range(ave_n):
						for j in range(self.env_n):
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
				for i in range(self.ave_n, self.env_n):
					actor = actors[i]
					critic = critics[i]
					if True: #replayMemory.size() > int(args["minibatch_size"]):
						# s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))
						
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

				global_step += 1
				update_event.clear()        # updating finished
				rolling_event.set()

###########################
#####    WORKER    ########
###########################

class Worker(object):
	# init
	def __init__(self, wid, brain, n, max_episode_len, batch_size, seed, noise):
		self.wid = wid
		self.env = ma.make_env("simple_tag")
		self.env.seed(int(seed))
		self.brain = brain
		self.agent_num = n
		self.max_episode_len = max_episode_len
		self.batch_size = batch_size
		self.noise = noise

	def work(self):
		global global_step_max, global_step, coord, global_queue, update_event, rolling_event

		while not coord.should_stop():
			s = self.env.reset()
			episode_reward = np.zeros((self.agent_num,))
			start = time.time()
			for stp in range(200):
				
				
				if not rolling_event.is_set():
					rolling_event.wait()
				self.env.render()
				actions = []
				for i in range(self.brain.actors):
					actor = self.barin.actors[i]
					actions.append(actor.act(state_input, noise[i]()).reshape(actor.action_dim,)) 

				s, r, done, s2 = self.env.step(actions)

				episode_reward += r

				global_queue.put([s, actions, r, d, s2])
				global_step += 1
				s = s2
				episode_reward += r

				
				print("wid:", self.wid, " working!")
				if stp == self.max_episode_len - 1 or global_queue.qsize() >= self.batch_size:
					showAveReward(self.wid, episode_reward, self.agent_num, ep, start)
					rolling_event.clear()
					update_event.set()

					if global_step >= global_step_max:
						coord.request_stop()
						break


def build_summaries(n):
	#episode_reward = tf.get_variable("episode_reward",[1,n])
	# record reward summay 
	# ave_reward = tf.Variable(0.)
	# good_reward = tf.Variable(0.)
	# episode_reward =   tf.Variable(0.)
	# tf.summary.scalar("Ave_Reward",ave_reward)
	# tf.summary.scalar("Good_Reward",good_reward)

	losses = [tf.Variable(0.) for i in range(n)]

	for i in range(n):
		tf.summary.scalar("Loss_Agent" + str(i), losses[i])
	
	#episode_ave_max_q = tf.Variable("episode_av_max_")
	#tf.summary.scalar("QMaxValue",episode_ave_max_q)
	#summary_vars = [episode_reward,episode_ave_max_q]
	# summary_vars = [ave_reward, good_reward]
	summary_vars = losses
	summary_ops = tf.summary.merge_all()
	return summary_ops, summary_vars

def getFromQueue():
	global global_queue
	s_batch, a_batch, r_batch, d_batch, s2_batch = [], [], [], [], []
	for i in range(global_queue.qsize()):
		data = global_queue.get()
		s_batch.append(data[0])
		a_batch.append(data[1])
		r_batch.append(data[2])
		d_batch.append(data[3])
		s2_batch.append(data[4])

	return s_batch, a_batch, r_batch, d_batch, s2_batch

class Controller(object):
	def __init__(self):
		self.update_event = update_event
		self.rolling_event = rolling_event

		self.update_event.clear()
		self.rolling_event.set()

		self.coord = tf.train.Coordinator()


def distributed_train(sess, env, args, actors, critics, noise, ave_n):

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
	
	worker_num = 2
	global update_event, rolling_event
	update_event, rolling_event = threading.Event(), threading.Event()
	update_event.clear()
	rolling_event.set()

	brain = Brain(actors, critics, ave_n, env.n)
	workers = [Worker(i, brain, env.n, 200, 64, 1234+i, noise) for i in range(worker_num)] 

	global global_step_max, global_step, coord, global_queue

	coord = tf.train.Coordinator()	

	global_queue = queue.Queue()

	threads = []

	
	global_step_max, global_step = 200*1000, 0

	
	for worker in workers:
		t = threading.Thread(target=worker.work, args=())
		t.start()
		threads.append(t)
	threads.append(threading.Thread(target=brain.update, args=()))
	threads[-1].start()
	
	coord.join(threads)
	# replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))
	"""
	for ep in range(int(args['max_episodes'])):

		start = time.time()

		s = env.reset()
		episode_reward = np.zeros((env.n,))
		#episode_av_max_q = 0

		s_batch, a_batch, r_batch, d_batch, s2_batch = getFromQueue(global_queue)

		for stp in range(int(args['max_episode_len'])):

					
			episode_reward += r
			#print(done)
			if stp == int(args["max_episode_len"])-1 or np.all(done) :
				
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
	"""

def saveModel(actor, i, pathToSave):
	actor.mainModel.save(pathToSave + str(i) + ".h5")

def saveWeights(actor, i, pathToSave):
	actor.mainModel.save_weights(pathToSave + str(i) + "_weights.h5")

def showReward(episode_reward, n, ep, start):
	reward_string = ""
	for re in episode_reward:
		reward_string += " {:5.2f} ".format(re)
	print ('|Episode: {:4d} | Time: {:2d} | Rewards: {:s}'.format(ep, int(time.time() - start), reward_string))

def showAveReward(wid, episode_reward, n, ep, start):
	reward_string = ""
	for re in episode_reward:
		reward_string += " {:5.2f} ".format(re / ep)
	print ('Worker: {:d} |Episode: {:4d} | Time: {:2d} | Rewards: {:s}'.format(wid, ep, int(time.time() - start), reward_string))


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


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
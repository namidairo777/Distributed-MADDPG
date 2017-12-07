import tensorflow as tf
import numpy as  np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Concatenate,Add,Activation,Lambda
from keras.optimizers import Adam
import keras.backend as K
# from keras.callbacks import EarlyStopoing, TensorBoard

class Brain(object):
	"""
	Brain consists of actor, old actor and critic.
	"""
	def __init__(self, sess, actor_state_dim, actor_action_dim, actor_lr, actor_tau, \
					critic_state_dim, critic_action_dim, critic_lr, critic_tau, gamma):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)

		# Initiation
		self.actor_state_dim = actor_state_dim
		self.critic_state_dim = critic_state_dim
		self.critic_action_dim = critic_action_dim

		# Construct Actor, old Actor, Critic network 
		self.actor, self.actor_weights, self.actor_input_state = self._build_actor_model()
		self.old_actor, self.old_actor_weights, self.actor_input_state = self._build_actor_model()
		self.critic, self.critic_input_actions, self.critic_input_state = self._build_critic_model()

		# critic update
		self.discounted_reward = Input(shape=(1, ))
		self.Q = self.critic.predict(self.critic_input_actions, self.critic_input_state)
		self.advantage = self.discounted_reward - self.Q

		self.critic_loss = tf.reduce_mean(tf.square(self.advantage))
		self.critic_train = tf.train.AdamOptimizer(criric_lr).minimize(self.critic_loss)

		# update old pi
		self.sample_actor = tf.squeeze(self.actor.output.sample(1), axis=0)
		


		# actor update
		# ratio 
		# self.advantage_for_clip = Input(shape=(1, ))
		ratio = self.actor.pi.prob(action) / (self.old_actor.prob(action) + 1e-5)

		surr = ratio * self.advantage

		self.actor_loss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.advantage))

		self.actor_train = tf.train.AdamOptimizer(actor_lr).minimize(self.actor_loss)
		self.sess.run(tf.global_variables_initializer())

	def _build_actor_model(self):
		input_obs = Input(shape=(self.actor_state_dim, ))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)

		h = Dense(300)(h)
		h = Activation('relu')(h)

		h = Dense(self.action_dim)(h)
		h = Activation('softmax')(h)

		pred = Dense(1, kernel_initializer='random_normal')(h)		
		model = Model(inputs=input_obs, outputs=pred)
		model.compile(optimizer='Adam', loss='categorical_crossentropy')
		
		return model, model.trainable_weights, input_obs


	def _build_critic_model(self):
		input_obs = Input(shape=(self.critic_state_dim,))
		input_actions = Input(shape=(self.critic_action_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)

		action_abs = Dense(64)(input_actions)
		temp1 = Dense(64)(h)

		h = Add()([temp1,action_abs])
		h = Activation('relu')(h)

		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		# , metrics=['mae']
		model.compile(optimizer='Adam',loss='mean_squared_error')
		
		return model, input_obs, input_actions

	def _update_old_actor(self):
		self.update_old_actor = self.critic.set_weights(np.asarray(self.actor.get_weights()))

	def update(self):
		# udpate old pi
		_update_old_actor()
		# get data from QUEUE
		# get data from 




	def choose_action(self):


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):


##################################
##### Actor network
##################################

class ActorNetwork(object):
	"""
	Implements actor network
	"""
	def __init__(self,sess,state_dim,action_dim,lr,tau):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.mainModel,self.mainModel_weights,self.mainModel_state = self._build_simple_model()
		self.targetModel,self.targetModel_weights,_ = self._build_simple_model()
		self.action_gradient = tf.placeholder(tf.float32,[None,self.action_dim])
		self.params_grad = tf.gradients(self.mainModel.output, self.mainModel_weights, self.action_gradient)
		grads = zip(self.params_grad,self.mainModel_weights)
		self.optimize = tf.train.AdamOptimizer(-self.lr).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

	def _build_hard_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(300)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('softmax')(h)
		
		# pred = Lambda(lambda h: tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample())(h)
		# pred = model.add(Lambda(ontHot(h))(h))

		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs

	def _build_simple_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('softmax')(h)
		
		# pred = Lambda(lambda h: tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample())(h)
		# pred = model.add(Lambda(ontHot(h))(h))

		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs


	# Choose an action
	def act(self,state,noise):
		act = self.mainModel.predict(state) + noise
		return act

	
	def predict_target(self,state):
		return self.targetModel.predict(state)

	def predict(self,state):
		return self.mainModel.predict(state)

	# Update target network parameter
	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		for i in range(len(wMain)):
			wTarget[i] = self.tau*wMain[i] + (1-self.tau)*wTarget[i]
		self.targetModel.set_weights(wTarget)

	# AdamOptimizer.minimize(TD loss)
	def train(self,state,action_grad):
		self.sess.run(self.optimize,feed_dict = {self.mainModel_state: state, self.action_gradient: action_grad})


##################################
##### Critic network
##################################
class CriticNetwork(object):
	def __init__(self,sess,num_agents,state_dim,action_dim,lr,tau,gamma):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		# print(self.action_dim)
		self.lr =  lr
		self.tau = tau
		self.num_agents = num_agents
		self.gamma  =  gamma
		self.mainModel,self.state,self.actions = self._build_simple_model()
		self.targetModel,_,_ = self._build_simple_model()
		self.action_grads  = tf.gradients(self.mainModel.output,self.actions)
		self.sess.run(tf.global_variables_initializer())

	# Simple Network model
	def _build_simple_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		action_abs = Dense(64)(input_actions)
		temp1 = Dense(64)(h)
		#action_abs = Activation('relu')(action_abs)
		#action_abs = BatchNormalization()(action_abs)
		h = Add()([temp1,action_abs])
		#h = Dense(64)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		# , metrics=['mae']
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions

	# Get action gradients for Actor to update from ys (Critic Q) / xs (Actions)
	def action_gradients(self,states,actions):
		return self.sess.run(self.action_grads,feed_dict={self.state: states, self.actions: actions})[0]

	# Update target network params
	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		self.targetModel.set_weights(self.tau*wMain + (1.0-self.tau)*wTarget)

	def predict_target(self, state, actions):
		return self.targetModel.predict([state,actions])

	def predict(self, state, actions):
		x = np.ndarray((actions.shape[1],self.action_dim))
		for j in range(actions.shape[1]):
			x[j] = np.concatenate([y[j] for y in actions])
		return self.mainModel.predict([state,x])

	# Batch training for Critic
	# train_on_batch(x, y)
	# x: training data
	# y: target data - 
	def train(self, state, actions, labels):
		return self.mainModel.train_on_batch([state,actions],labels)
		#return self.predict(state,actions)


"""
eP 4750 network architecture

Actor
# Network architecture
	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('softmax')(h)
		#pred = tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample()
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs


Critic 
def _build_hard_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		action_abs = Dense(300)(input_actions)
		temp1 = Dense(300)(h)
		#action_abs = Activation('relu')(action_abs)
		#action_abs = BatchNormalization()(action_abs)
		h = Add()([temp1,action_abs])
		#h = Dense(64)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions



"""
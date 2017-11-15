import tensorflow as tf
import numpy as  np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Concatenate,Activation
from keras.optimizers import Adam
import keras.backend as K

class ActorNetwork(object):
	"""
	Implements actor network
	"""
	def __init__(self, sess, state_dim, action_dim, lr, tau):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.mainModel, self.mainModel_weights, self.mainModel_state = self._build_model()
		self.targetModel, self.targetModel_weights, _ = self._build_model()
		# fed action gradients [none, action_dim]
		self.action_gradient = tf.placeholder(tf.float32, [None,self.action_dim])
		# Parameter gradients (ys, xs, grad_ys)
		self.params_grad = tf.gradients(self.mainModel.output, self.mainModel_weights, -self.action_gradient)
		grads = zip(self.params_grad, self.mainModel_weights)
		# apply_gradients(grads and vars)
		self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

	# Build network
	def _build_model(self):
		# input = tf.placeholder(tf.float32, [none, self.state_dim])
		input_obs = Input(shape=(self.state_dim, ))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('softmax')(h)

		model = Model(inputs=input_obs, outputs=pred)
		model.compile(optimizer='Adam', loss='categorical_crossentropy')
		return model, model.trainable_weights, input_obs

	# Main model to predcict action with noise 
	def act(self, state, noise):
		act = self.mainModel.predict(state) + noise
		return act

	# target model
	def predict_target(self, state):
		return self.targetModel.predict(state)

	# without noise
	def predict(self, state):
		return self.mainModel.predict(state)

	# update target network, default tau = 0.01
	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		self.targetModel.set_weights(self.tau*wMain + (1.0-self.tau)*wTarget)

	# To optimize network
	def train(self, state, action_grad):
		self.sess.run(self.optimize, feed_dict = {self.mainModel_state: state,  self.action_gradient: action_grad})


class CriticNetwork(object):
	def __init__(self, sess, num_agents, state_dim, action_dim, lr, tau, gamma):
		self.sess = sess
		K.set_session(sess)
		K.set_learning_phase(1)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr =  lr
		self.tau = tau
		self.num_agents = num_agents
		self.gamma  =  gamma
		self.mainModel, self.state, self.actions = self._build_model()
		self.targetModel, _, _ = self._build_model()
		# gradients(Q, actions)
		self.action_grads  = tf.gradients(self.mainModel.output, self.actions)
		self.sess.run(tf.global_variables_initializer())

	def _build_model(self):
		input_obs = Input(shape=(self.state_dim, ))
		input_actions = Input(shape=(self.action_dim, ))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		action_abs = Dense(64)(input_actions)
		action_abs = Activation('relu')(action_abs)
		action_abs = BatchNormalization()(action_abs)
		# concatenate() is used to merge layers
		h = Concatenate()([h, action_abs])
		h = Dense(64)(h)
		h = Activation('relu')(h)
		h = BatchNormalization()(h)
		# random_uniform?
		pred = Dense(1, kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs, input_actions], outputs=pred)
		# Mean_squared_error for one output
		model.compile(optimizer='Adam', loss='mean_squared_error')
		return model, input_obs, input_actions

	# return action gradients for main model
	def action_gradients(self, states, actions):
		return self.sess.run(self.action_grads, feed_dict={self.state: states,  self.actions: actions})[0]

	def update_target(self):
		wMain =  np.asarray(self.mainModel.get_weights())
		wTarget = np.asarray(self.targetModel.get_weights())
		self.targetModel.set_weights(self.tau*wMain + (1.0-self.tau)*wTarget)

	def predict_target(self,  state,  actions):
		return self.targetModel.predict([state, actions])

	def predict(self,  state,  actions):
		x = np.ndarray((actions.shape[1], self.action_dim))
		for j in range(actions.shape[1]):
			x[j] = np.concatenate([y[j] for y in actions])
		return self.mainModel.predict([state, x])

	# train on batch?
	def train(self,  state,  actions,  labels):
		self.mainModel.train_on_batch([state, actions], labels)
		#return self.predict(state, actions)
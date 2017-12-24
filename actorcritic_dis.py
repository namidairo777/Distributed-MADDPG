import numpy as  np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Concatenate,Add,Activation,Lambda
from keras.optimizers import Adam
from keras import initializers
import keras.backend as K
import tensorflow as tf
# from keras.callbacks import EarlyStopoing, TensorBoard

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
		self.mainModel,self.mainModel_weights,self.mainModel_state = self._build_baseline1_model()
		self.mainModel._make_predict_function()
		self.targetModel,self.targetModel_weights,_ = self._build_baseline1_model()
		self.targetModel._make_predict_function()
		self.action_gradient = tf.placeholder(tf.float32,[None,self.action_dim])
		self.params_grad = tf.gradients(self.mainModel.output, self.mainModel_weights, self.action_gradient)
		grads = zip(self.params_grad,self.mainModel_weights)
		self.optimize = tf.train.AdamOptimizer(-self.lr).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())
		# self.default_graph = tf.get_default_graph()
		# self.default_graph.finalize()
	
	
	def _build_baseline_model(self):
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
		# pred = Lambda(lambda h: tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample())(h)
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs

	def _build_baseline1_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(300)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('tanh')(h)
		
		# pred = Lambda(lambda h: tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample())(h)
		# pred = model.add(Lambda(ontHot(h))(h))
		# pred = Lambda(lambda h: tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample())(h)
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs

	def _build_simple_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = Dense(64)(h)
		h = Activation('relu')(h)		
		h = Dense(self.action_dim, kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h)
		pred = Activation('tanh')(h)
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
		self.mainModel,self.state,self.actions = self._build_maddpg_model()
		self.mainModel._make_predict_function()
		self.mainModel._make_train_function()
		self.targetModel,_,_ = self._build_maddpg_model()
		self.targetModel._make_predict_function()
		self.action_grads  = tf.gradients(self.mainModel.output,self.actions)
		self.sess.run(tf.global_variables_initializer())
		# self.default_graph = tf.get_default_graph()
		# self.default_graph.finalize()

	def _build_maddpg_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		temp1 = Dense(300)(h)
		action_abs = Dense(300)(input_actions)
		# action_abs = BatchNormalization()(action_abs)
		h = Add()([temp1,action_abs])
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions

	def _build_baseline_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		
		temp_obs = Dense(400)(input_obs)
		obs1 = Activation('relu')(temp_obs)
		obs2 = Dense(300)(obs1)
		actions1 = Dense(300)(input_actions)

		h = Add()([obs2, actions1])
		h = Dense(300)(h)
		h = Activation('relu')(h)

		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
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
	
	def get_loss(self, state, actions, labels):
		return self.mainModel.test_on_batch([state,actions],labels)
		#return self.predict(state,actions) 


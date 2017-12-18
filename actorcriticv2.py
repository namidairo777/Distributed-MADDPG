import tensorflow as tf
import numpy as  np
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Concatenate,Add,Activation,Lambda
from keras.optimizers import Adam
from keras import initializers
import keras.backend as K
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
		self.mainModel,self.mainModel_weights,self.mainModel_state = self._build_hard2_model()
		self.mainModel._make_predict_function()
		self.targetModel,self.targetModel_weights,_ = self._build_hard2_model()
		self.targetModel._make_predict_function()
		self.action_gradient = tf.placeholder(tf.float32,[None,self.action_dim])
		self.params_grad = tf.gradients(self.mainModel.output, self.mainModel_weights, self.action_gradient)
		grads = zip(self.params_grad,self.mainModel_weights)
		self.optimize = tf.train.AdamOptimizer(-self.lr).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())
		# self.default_graph = tf.get_default_graph()
		# self.default_graph.finalize()
	
	# Network architecture
	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		h = Dense(128)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(128)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		h = Dense(self.action_dim)(h)
		pred = Activation('softmax')(h)
		#pred = tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=h).sample()
		model = Model(inputs=input_obs,outputs=pred)
		model.compile(optimizer='Adam',loss='categorical_crossentropy')
		return model,model.trainable_weights,input_obs

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

	def _build_hard2_model(self):
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
		self.mainModel,self.state,self.actions = self._build_hard3_model()
		self.mainModel._make_predict_function()
		self.mainModel._make_train_function()
		self.targetModel,_,_ = self._build_hard3_model()
		self.targetModel._make_predict_function()
		self.action_grads  = tf.gradients(self.mainModel.output,self.actions)
		self.sess.run(tf.global_variables_initializer())
		# self.default_graph = tf.get_default_graph()
		# self.default_graph.finalize()

	# Network architecture
	def _build_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(128)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		action_abs = Dense(128)(input_actions)
		temp1 = Dense(128)(h)
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

	def _build_hard2_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(400)(input_obs)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		# action_abs = Dense(300)(input_actions)
		# temp1 = Dense(300)(h)
		#action_abs = Activation('relu')(action_abs)
		#action_abs = BatchNormalization()(action_abs)
		h = Concatenate(axis=-1)([h,input_actions])
		h = Dense(300)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions

	def _build_hard3_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		temp_obs = Dense(400)(input_obs)
		obs = Activation('relu')(temp_obs)
		temp_actions = Dense(400)(input_actions)
		actions =  Activation('relu')(temp_actions)

		#h = BatchNormalization()(h)
		# action_abs = Dense(300)(input_actions)
		# temp1 = Dense(300)(h)
		#action_abs = Activation('relu')(action_abs)
		#action_abs = BatchNormalization()(action_abs)
		h = Add()([obs,actions])
		h = Dense(300)(h)
		h = Activation('relu')(h)
		#h = BatchNormalization()(h)
		pred = Dense(1,kernel_initializer='random_uniform')(h)
		model = Model(inputs=[input_obs,input_actions],outputs=pred)
		model.compile(optimizer='Adam',loss='mean_squared_error')
		return model,input_obs,input_actions

	# Simple Network model
	def _build_simple_model(self):
		input_obs = Input(shape=(self.state_dim,))
		input_actions = Input(shape=(self.action_dim,))
		h = Dense(64)(input_obs)
		h = Activation('relu')(h)
		h = Concatenate(axis=-1)([h,input_actions])
		h = Dense(64)(h)
		h = Activation('relu')(h)
		pred = Dense(1, kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h)
		# pred = Dense(1,initializers.RandomUniform(minval=-3e-3, maxval=3e-3))(h)
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
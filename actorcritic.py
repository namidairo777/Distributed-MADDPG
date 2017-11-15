import tensorflow as tf
import numpy as np
import tflearn
import random

class ActorNetwork(object):
	"""
	This class defines the Actor Network for DDPG. Input is the observation from the environment.
	Output is the action decided by a deterministic policy
	"""
	def __init__(self,sess,n,observation_dim,action_dim,learning_rate,target_update_param):
		self.sess = sess
		self.input_dim = observation_dim
		self.output_dim = action_dim
		self.lr = learning_rate
		self.tau = target_update_param
		
		self.input,self.gumbel_output,self.argmax_output = self.initializeActorNetwork()
		self.network_params = tf.trainable_variables()

		self.target_input,self.gumbel_target_output,self.argmax_target_output = self.initializeActorNetwork()
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]
		
		self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1.0-self.tau)) for i in range(len(self.target_network_params))]
		self.action_gradient = tf.placeholder(tf.float32,[None,self.output_dim])
		
		self.actor_gradient = tf.gradients(self.gumbel_output,self.network_params,-self.action_gradient)
		self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradient,self.network_params))
		
		self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
	
	def initializeActorNetwork(self):
		inputs = tflearn.input_data(shape=[None,self.input_dim])
		net = tflearn.fully_connected(inputs,64)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		net = tflearn.fully_connected(net,64)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)
		w_init = tflearn.initializations.uniform(minval = -0.003, maxval = 0.003)
		net = tflearn.fully_connected(net,self.output_dim,activation='softmax',weights_init=w_init)
		gumbel_out = tf.contrib.distributions.RelaxedOneHotCategorical(0.1,probs=net).sample()
		argmax_out = tf.one_hot(tf.argmax(net),self.output_dim)
		#scaled_out = tf.multiply(out,self.action_bound)
		return inputs,gumbel_out,argmax_out

	def train(self,inputs,action_gradient):
		self.sess.run(self.optimize,feed_dict = {self.input: inputs, self.action_gradient: action_gradient})

	def predict(self,inputs):
		return self.sess.run(self.argmax_output,feed_dict = {self.input: inputs})

	def predictTarget(self,target_inputs):
		return self.sess.run(self.argmax_target_output,feed_dict = {self.target_input: target_inputs})

	def updateTargetNetwork(self):
		self.sess.run(self.update_target_network_params)

	def getNumTrainableVars(self):
		return self.num_trainable_vars


class CriticNetwork(object):
	"""
	This class implements the critic in DDPG, which learns the action-value function Q, and provides action gradients to the actor
	"""
	def __init__(self,sess,n,observation_dim,action_dim,learning_rate,target_update_param,gammaVar,num_actor_vars):
		self.sess = sess
		self.input_dim = observation_dim
		self.output_dim = action_dim
		self.lr = learning_rate
		self.tau = target_update_param
		self.gamma = gammaVar
		self.input,self.action,self.output = self.initializeCriticNetwork()
		self.network_params = tf.trainable_variables()[num_actor_vars:]
		self.target_input,self.target_action,self.target_output = self.initializeCriticNetwork()
		self.target_network_params = tf.trainable_variables()[(len(self.network_params)+num_actor_vars):]
		self.update_target_network_params = [self.target_network_params[i].assign(tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1.0-self.tau)) for i in range(len(self.target_network_params))]
		self.predictedQ = tf.placeholder(tf.float32,[None,1])
		self.loss = tflearn.mean_square(self.predictedQ,self.output)
		self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		self.action_gradient = tf.gradients(self.output,self.action)
	
	def initializeCriticNetwork(self):
		inputs = tflearn.input_data(shape = [None,self.input_dim])
		action = tflearn.input_data(shape = [None,self.output_dim])
		net = tflearn.fully_connected(inputs,64)
		net = tflearn.layers.normalization.batch_normalization(net)
		net = tflearn.activations.relu(net)

		temp1 = tflearn.fully_connected(net,64)
		temp2 = tflearn.fully_connected(action,64)
		net = tflearn.activation(tf.matmul(net,temp1.W)+tf.matmul(action,temp2.W)+temp2.b,activation = 'relu')
		
		w_init = tflearn.initializations.uniform(minval = -0.003,maxval = 0.003)
		out = tflearn.fully_connected(net,1,weights_init = w_init)
		return inputs,action,out

	def train(self,inputs,action,predictedQ):
		return self.sess.run([self.output,self.optimize],feed_dict = {self.input: inputs, self.action: action, self.predictedQ: predictedQ})

	def predict(self,inputs,action):
		return self.sess.run(self.output,feed_dict = {self.input: inputs, self.action: action})

	def predictTarget(self,inputs,action):
		return self.sess.run(self.target_output, feed_dict = {self.target_input: inputs, self.target_action: action})

	def actionGradients(self,inputs,actions):
		return self.sess.run(self.action_gradient,feed_dict = {self.input: inputs, self.action: actions})

	def updateTargetNetwork(self):
		self.sess.run(self.update_target_network_params)

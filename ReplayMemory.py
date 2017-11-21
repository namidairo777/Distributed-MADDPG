from collections import deque
import random
import numpy as np

class ReplayMemory(object):

	def __init__(self,max_size = 100000,random_seed = 123):
		self.max_size = max_size
		self.buffer = deque(maxlen = self.max_size)
		random.seed(random_seed)

	def add(self,state,action,reward,done,next_state):
		exp = (state,action,reward,done,next_state)
		self.buffer.append(exp) 
	
	def size(self):
		return len(self.buffer)

	def miniBatch(self,batch_size):
		miniBatch = random.sample(self.buffer,min(len(self.buffer),batch_size))
		state_batch = np.array([_[0] for _ in miniBatch])
		action_batch = np.array([_[1] for _ in miniBatch])
		reward_batch = np.array([_[2] for _ in miniBatch])
		done_batch = np.array([_[3] for _ in miniBatch])
		next_state_batch = np.array([_[4] for _ in miniBatch])
		return state_batch,action_batch,reward_batch,done_batch,next_state_batch

	def clear(self):
		self.buffer.clear()
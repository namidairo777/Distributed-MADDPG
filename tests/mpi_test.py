import tensorflow as tf
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import gym
import time

env = gym.make("CartPole-v1")
s = env.reset()

rewards = 0
with tf.Session() as sess:
	for i in range(20):
		env.render()
		time.sleep(0.5)
		action = env.action_space.sample()
		s2, r, d, _ = env.step(action)
		
		rewards += r
		s = s2
	print(sess.run("sess"))



print("Rank ", rank, "Reward: ", rewards)

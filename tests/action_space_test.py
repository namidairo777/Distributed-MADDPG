import make_env as ma
import time
import random

env = ma.make_env("simple_tag")
s = env.reset()
print(env.observation_space)
for i in range(1000):
	actions = []
	for j in range(env.n):
		actions.append([0, 0, 0, 0.5, 0])
	s2, r, d, _ = env.step(actions)
	env.render()
	s = s2
	time.sleep(0.05)
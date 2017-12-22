from actorcriticv2 import ActorNetwork
import multiprocessing as mp
import tensorflow as tf

# from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#s et_session(tf.Session(config=config))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

model2 = ActorNetwork(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)), 3, 4, 0.1, 0.88)

class Worker():
	def __init__(self):
		self.output = "tensorflow"

	def train(self, weights):
		print("training")
		#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True)) as sess:	

		model2.update_target()
		model2.mainModel.set_weights(weights)
			
		return "Work Good" + self.output

worker = Worker()
	
if __name__ == '__main__':
	
	model = ActorNetwork(sess, 3, 4, 0.1, 0.88)
	mp.set_start_method('spawn')
	model1 = ActorNetwork(tf.Session(), 3, 4, 0.1, 0.88)
	pool = mp.Pool(processes=6)
	weights = model1.mainModel.get_weights()	
	out = []
	#workers = [Worker(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=True))) for i in range(4)]
	
	for i in range(3):
		jobs = [pool.apply_async(worker.train, args=(weights, )) for j in range(4)]
		data = []
		for job in jobs:
			data.append(job.get())
		out.append(data)
	
	#print("asd")
	print(out)

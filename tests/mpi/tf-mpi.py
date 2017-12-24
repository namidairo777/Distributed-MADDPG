from mpi4py import MPI
#import tensorflow as tf
import numpy as np
import time

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def worker(wid):
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
		hello = tf.constant("Worker" + str(wid) + "is working!!")
		print(sess.run(hello))

class Master(object):
	def __init__(self, sess, comm):
		self.sess = sess
		self.comm = comm

	def sendData(self):
		for i in range(1, self.comm.Get_size()):
			self.comm.isend("good job!", dest=i, tag=11)
		time.sleep(1)

	def getBatch(self, data):
		print("thank you for data {}".format(data))



if __name__ == "__main__":
	

	if rank == 0:
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		replay = []
		batch_signals = [comm.irecv(source=i, tag=11) for i in range(1, size)]
		for i in range(len(batch_signals)):
			if batch_signals[i]:
				replay.append(batch_signals[i].wait())
				print("recieve batch from ", i+1)

		print(replay)
		# update 


		master = Master("sdf", comm)

		while False:
			master.sendData()
			for i in range(1, size):
				data = comm.recv(source=i, tag=11)
				#data = req.wait()
				master.getBatch(data)
		master.sendData()


	else:
		print(comm)
		while False:
			# recieve weights data

			data = comm.recv(source=0, tag=11)
			# time.sleep(5)
			timestep = time.time()
			print("Worker {} recieve data from master: {} at {}".format(rank, data, timestep))

			# send batch data async
			batch_data = np.array([0.35, 0.85, 0.35])
			comm.send(batch_data, dest=0, tag=11)
			# req.wait()
		#req = comm.irecv(source=0, tag=11)
		
		data = comm.irecv(source=0, tag=11)
		#data = req.wait()
		# print("test")

		timestep = time.time()
		if data.Get_status():
			# get weights and update
			print("Async data")
			print("Worker {} recieve weights from master: {} at {}".format(rank, data.wait(), timestep))

		# noraml work: get batch data
		print("Worker {} is doing normal work at {}".format(rank, timestep))
		# comm.isend("batch" + str(rank), dest=0, tag=11)
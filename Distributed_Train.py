import numpy as np
import gym, threading, queue
import make_env as ma
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
from keras.callbacks import TensorBoard
import make_env
import multiprocessing as mp
import time, os
from actorcriticv2 import ActorNetwork
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
# from actorcritic import ActorNetwork,CriticNetwork

###########################
#####     BRAIN    ########
###########################


###########################
#####    WORKER    ########
###########################

class Worker(object):
    # init
    def __init__(self, wid, n, max_episode_len, batch_size, seed, noise):
        self.wid = wid
        self.env = make_env.make_env("simple_tag")
        print("Initiate worker ", wid)
        self.env.seed(int(seed))
        self.agent_num = n
        self.max_episode_len = max_episode_len
        self.batch_size = batch_size
        self.noise = noise
        self.actors = []

    def work(self, env):

        print("Worker ", self.wid, "starts working")
        s = env.reset()

        batch_data = []

        for stp in range(self.max_episode_len):
            
            actions = []
            for i in range(self.agent_num):
                      # print("Taking actions")
                actor = self.actors[i]    

                state_input = np.reshape(s[i],(-1,actor.state_dim))
                
                actions.append(actor.act(state_input, self.noise[i]()).reshape(actor.action_dim,)) 
                    
            s2, r, done, _ = env.step(actions)

            batch_data.append((s, actions, r, done, s2))

            s = s2

            # if stp == self.max_episode_len - 1:
        return batch_data  

def build_summaries(n):
    losses = [tf.Variable(0.) for i in range(n)]

    for i in range(n):
        tf.summary.scalar("Loss_Agent" + str(i), losses[i])
    summary_vars = losses
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

def getFromQueue():
    s_batch, a_batch, r_batch, d_batch, s2_batch = [], [], [], [], []
    for i in range(global_queue.qsize()):
        data = global_queue.get()
        s_batch.append(data[0])
        a_batch.append(data[1])
        r_batch.append(data[2])
        d_batch.append(data[3])
        s2_batch.append(data[4])

    return s_batch, a_batch, r_batch, d_batch, s2_batch

class Controller(object):
    def __init__(self):
        self.update_event = update_event
        self.rolling_event = rolling_event

        self.update_event.clear()
        self.rolling_event.set()

        self.coord = tf.train.Coordinator()

def get_batch(worker, actors):
    return worker.work()

class SampleA(object):
    def __init__(self, i, sess):
        self.wid = i
        self.model = ActorNetwork(sess, 1, 2, 0.1, 0.9)
    def out(self, i):
        print(self.wid)
        self.model.update_target()



def work(j):
    
    global workers
    env = make_env.make_env("simple_tag")
    s = env.reset()
    batch_data = []
    for stp in range(5):
        actions = []
        for i in range(3):
            actor = workers[j][i]    
            state_input = np.reshape(s[i],(-1,actor.state_dim))
            actions.append(actor.act(state_input, exploration_noise[i]()).reshape(actor.action_dim,)) 
        s2, r, done, _ = env.step(actions)
        batch_data.append((s, actions, r, done, s2))
        s = s2
    return batch_data  


def distributed_train(sess, env, args, actors, critics, noise, ave_n):

    worker_num = 4
    #########
    # Worker session
    #
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    worker_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    
    global workers

    workers = [[] for i in range(worker_num)]
    for actor in actors:
        for worker in workers:
            worker.append(ActorNetwork(worker_sess, actor.state_dim, actor.action_dim, actor.lr, actor.tau))
    #######################
    print(len(workers), len(workers[0]))

    global exploration_noise
    exploration_noise =[] 
    
    for actor in actors:
        exploration_noise.append(OUNoise(mu = np.zeros(actor.action_dim)))
        actor.update_target()
    for critic in critics:
        critic.update_target()
    

    pool = mp.Pool(processes=mp.cpu_count()-1)

    replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

    for timestep in range(int(args['max_episodes'] * args['max_episode_len'])):

        start = time.time()

        # print(workers[0].work())
        # jobs = [pool.apply_async(sample.out, ()) for sample in samples]

        jobs = [pool.apply_async(work, args=(j, )) for j in range(len(workers))]
        
        # res = pool.map(samples[0].out, [1,2,3])
        #time.sleep(10)
        
        for job in jobs:
            data = job.get()
            for item in data:
                (s, a, r, d, s2) = item
                print(item)
                # replayMemory.add(s,a,r,done,s2)
        
        sleep(10)
        #losses = []
        action_dims_done = 0


        # MADDPG Adversary Agent            
        for i in range(ave_n):
            actor = actors[i]
            critic = critics[i]
            
            s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(int(args['minibatch_size']))
            a = []
            for j in range(ave_n):
                state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
                a.append(actors[j].predict_target(state_batch_j))
            
            a_temp = np.transpose(np.asarray(a),(1,0,2))
            
            a_for_critic = np.asarray([x.flatten() for x in a_temp])
            s2_batch_i = np.asarray([x for x in s2_batch[:,i]]) # Checked till this point, should be fine.
            
            targetQ = critic.predict_target(s2_batch_i,a_for_critic) # Should  work, probably
            yi = []
            for k in range(int(args['minibatch_size'])):
                if d_batch[:,i][k]:
                    yi.append(r_batch[:,i][k])
                else:
                    yi.append(r_batch[:,i][k] + critic.gamma*targetQ[k])
            s_batch_i = np.asarray([x for x in s_batch[:,i]])
            
            critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch[:, 0: ave_n, :]]),np.asarray(yi))
            #losses.append(loss)
            
            actions_pred = []
            for j in range(ave_n):
                state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
                actions_pred.append(actors[j].predict(state_batch_j)) # Should work till here, roughly, probably
            a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
            a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
            s_batch_i = np.asarray([x for x in s_batch[:,i]])
            grads = critic.action_gradients(s_batch_i,a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
            actor.train(s_batch_i,grads)
            action_dims_done = action_dims_done + actor.action_dim
        # Only DDPG agent
        
        for i in range(ave_n, env.n):
            actor = actors[i]
            critic = critics[i]
            s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))
            s_batch_i = np.asarray([x for x in s_batch[:,i]])
            action = np.asarray(actor.predict_target(s_batch_i))
            
            action_for_critic = np.asarray([x.flatten() for x in action])
            s2_batch_i = np.asarray([x for x in s2_batch[:, i]])
            targetQ = critic.predict_target(s2_batch_i, action_for_critic)
            y_i = []
            for k in range(int(args['minibatch_size'])):
                # If ep is end
                if d_batch[:, i][k]:
                    y_i.append(r_batch[:, i][k])
                else:
                    y_i.append(r_batch[:, i][k] + critic.gamma * targetQ[k])
            # state batch for agent i
            s_batch_i= np.asarray([x for x in s_batch[:, i]])
            critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))
            #losses.append(loss)
            action_for_critic_pred = actor.predict(s2_batch_i)
            gradients = critic.action_gradients(s_batch_i, action_for_critic_pred)[:, :]
            actor.train(s_batch_i, gradients)
        
        for i in range(0, env.n):
            actor = actors[i]
            critic = critics[i]
            actor.update_target()
            critic.update_target()
        
        episode_reward += r
        
        if timestep % int(args["max_episode_len"]) == 0:
            print("timestep: ", timestep)
            print("time: ", time.time() - start)
            # showReward(episode_reward, env.n, ep, start)
            
        """
        # save model
        if ep % 50 == 0 and ep != 0:
            print("Starting saving model weights every 50 episodes")
            for i in range(env.n):
                # saveModel(actors[i], i, args["modelFolder"])
                saveWeights(actors[i], i, args["modelFolder"])
            print("Model weights saved")

        if ep % 200 == 0 and ep != 0:
            directory = args["modelFolder"] + "ep" + str(ep) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            print("Starting saving model weights to folder every 200 episodes")
            for i in range(env.n):
                # saveModel(actors[i], i, args["modelFolder"])
                saveWeights(actors[i], i, directory)
            print("Model weights saved to folder")
        """

        # print("Cost Time: ", int(time.time() - start), "s")




def saveModel(actor, i, pathToSave):
    actor.mainModel.save(pathToSave + str(i) + ".h5")

def saveWeights(actor, i, pathToSave):
    actor.mainModel.save_weights(pathToSave + str(i) + "_weights.h5")

def showReward(episode_reward, n, ep, start):
    reward_string = ""
    for re in episode_reward:
        reward_string += " {:5.2f} ".format(re)
    print ('|Episode: {:4d} | Time: {:2d} | Rewards: {:s}'.format(ep, int(time.time() - start), reward_string))

def showAveReward(wid, episode_reward, n, ep, start):
    reward_string = ""
    for re in episode_reward:
        reward_string += " {:5.2f} ".format(re / ep)
    global global_step
    print ('Global step: {:6.0f} | Worker: {:d} | Rewards: {:s}'.format(global_step, wid, reward_string))


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

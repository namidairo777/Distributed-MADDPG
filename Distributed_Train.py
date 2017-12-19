import numpy as np
import gym, threading, queue
import make_env as ma
import tensorflow as tf
import random
from ReplayMemory import ReplayMemory
from keras.callbacks import TensorBoard
# from dmaddpg import Brain, Worker
import time, os
# from actorcritic import ActorNetwork,CriticNetwork

###########################
#####     BRAIN    ########
###########################

class Brain(object):
    def __init__(self, modelFolder):
        self.actors = None
        self.critics = None
        self.ave_n = None
        self.env_n = None
        self.modelFolder = modelFolder

    def update(self):
        
        global global_step, global_step_max
        while not coord.should_stop():
            if global_step < global_step_max: 
                update_event.wait()
                # print("Brain working!")
                #global global_queue
                s_batch, a_batch, r_batch, d_batch, s2_batch = [], [], [], [], []
                for i in range(global_queue.qsize()):
                    data = global_queue.get()
                    s_batch.append(data[0])
                    a_batch.append(data[1])
                    r_batch.append(data[2])
                    d_batch.append(data[3])
                    s2_batch.append(data[4])

                s_batch = np.array(s_batch)
                a_batch = np.array(a_batch)
                r_batch = np.array(r_batch)
                d_batch = np.array(d_batch)
                s2_batch = np.array(s2_batch)
                # print("batch size:", s_batch.shape, s2_batch.shape)
                action_dims_done = 0
                for i in range(self.ave_n):
                    actor = self.actors[i]
                    critic = self.critics[i]
                    if True: 
                        a = []
                        for j in range(self.ave_n):
                            state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
                            a.append(self.actors[j].predict_target(state_batch_j))
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
                        
                        critic.train(s_batch_i,np.asarray([x.flatten() for x in a_batch[:, 0: self.ave_n, :]]),np.asarray(yi))
                        
                        actions_pred = []
                        for j in range(self.ave_n):
                            state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
                            actions_pred.append(self.actors[j].predict(state_batch_j)) # Should work till here, roughly, probably
                        a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
                        a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
                        s_batch_i = np.asarray([x for x in s_batch[:,i]])
                        grads = critic.action_gradients(s_batch_i,a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
                        actor.train(s_batch_i,grads)
                        # actor.update_target()
                        # critic.update_target()
                    action_dims_done = action_dims_done + actor.action_dim
                
                # Only DDPG agent
                for i in range(self.ave_n, self.env_n):
                    actor = self.actors[i]
                    critic = self.critics[i]
                    if True:            
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

                        s_batch_i= np.asarray([x for x in s_batch[:, i]])
                        critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))
                        action_for_critic_pred = actor.predict(s2_batch_i)
                        gradients = critic.action_gradients(s_batch_i, action_for_critic_pred)[:, :]
                        actor.train(s_batch_i, gradients)
                        
                for i in range(self.env_n):
                    actor = self.actors[i]
                    critic = self.critics[i]
                    actor.update_target()
                    critic.update_target()

                global_step += 1

                if global_step % (100*50) == 0 and global_step != 0:
                    directory = self.modelFolder + "ep" + str(global_step) + "/"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        print("Starting saving model weights to folder every 200 episodes")
                    for i in range(self.env_n):
                        # saveModel(actors[i], i, args["modelFolder"])
                        saveWeights(self.actors[i], i, directory)
                        print("Model weights saved to folder")

                update_event.clear()        # updating finished
                rolling_event.set()

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
        self.brain = brain
        self.agent_num = n
        self.max_episode_len = max_episode_len
        self.batch_size = batch_size
        self.noise = noise

    def work(self):
        global global_step_max, global_step
        
        while not coord.should_stop():
            s = self.env.reset()
            episode_reward = np.zeros((self.agent_num,))
            start = time.time()
            # print("env", s[0])
            buffer_s, buffer_a, buffer_r = [], [], []
            for stp in range(200):
                if not rolling_event.is_set():
                    rolling_event.wait()
                   
                # self.env.render()
                actions = []
                global graph
                with graph.as_default():
                # print("s0:", s[0])
                    for i in range(self.agent_num):
                        # print("Taking actions")
                        actor = self.brain.actors[i]    

                        # print("wid:", self.wid, " actor!", i)
                        state_input = np.reshape(s[i],(-1,actor.state_dim))
                        # print(state_input)
                        
                        actions.append(actor.act(state_input, self.noise[i]()).reshape(actor.action_dim,)) 
                    
                    s2, r, done, _ = self.env.step(actions)

                    episode_reward += r

                    if stp == self.max_episode_len - 1:
                        Q = self.

                if global_queue.qsize() < self.batch_size:
                    global_queue.put([s, actions, r, done, s2])
                # global_step += 1
                s = s2
                episode_reward += r

                if stp == self.max_episode_len - 1:
                    if self.wid == 0:
                        showAveReward(self.wid, episode_reward, self.agent_num, stp, start)
                    break
                
                if global_queue.qsize() > self.batch_size - 1:
                    
                    rolling_event.clear()
                    update_event.set()
                    

                    if global_step >= global_step_max:
                        coord.request_stop()
                        break


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


def distributed_train(sess, env, args, actors, critics, noise, ave_n):

    global graph, global_queue, update_event, rolling_event, global_step_max, global_step, coord, brain
    graph = tf.get_default_graph()
    global_queue = queue.Queue()
    update_event, rolling_event = threading.Event(), threading.Event()
    global_step_max, global_step = 200*10000, 0
    coord = tf.train.Coordinator()
    brain = Brain(args["modelFolder"])
    
    for actor in actors:
        actor.update_target()
    for critic in critics:
        critic.update_target()
    
    worker_num = 4
    global update_event, rolling_event
    update_event.clear()
    rolling_event.set()
    
    brain.actors = actors
    brain.critics = critics
    brain.ave_n = ave_n
    brain.env_n = env.n

    workers = [Worker(i, env.n, 200, 64, 1234+i, noise) for i in range(worker_num)] 

    global_queue = queue.Queue()

    threads = []

    for worker in workers:
        t = threading.Thread(target=worker.work, args=())
        threads.append(t)
    threads.append(threading.Thread(target=brain.update, args=()))
    
    for t in threads:
        t.start()
        #time.sleep(0.2)

    # print("before worker")
    coord.join(threads)
    

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

import tensorflow as tf
from gym import wrappers
import make_env
import numpy as np
#import random
#from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcriticv2 import ActorNetwork,CriticNetwork
#from actorcriticv1 import Brain, Worker
# from Train import train
# from Distributed_Train import *
import argparse
from keras.models import load_model
import os
import threading, queue, time
import multiprocessing as mp


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
                        
                        # actor.update_target()
                        # critic.update_target()
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

    
    
    # callbacks = []
    # train_names = ['train_loss', 'train_mae']
    # callback = TensorBoard(args['summary_dir'])
     

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


    threads = []

    for worker in workers:
        t = mp.Process(target=worker.work, args=())
        #t = threading.Thread(target=worker.work, args=())
        threads.append(t)
    threads.append(mp.Process(target=brain.update, args=()))
    
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



def main(args):

    if not os.path.exists(args["modelFolder"]):
        os.makedirs(args["modelFolder"])
    if not os.path.exists(args["summary_dir"]):
        os.makedirs(args["summary_dir"])


    #with tf.device("/gpu:0"):
    # MADDPG for Ave Agent
    # DDPG for Good Agent
    with tf.Session() as sess:

        env  = make_env.make_env('simple_tag')

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        #with tf.device('/cpu:0'):
            #if args["runTest"]:
                #run()
                #import sys
                #sys.exit("test over!")

        # Calculate good and ave agents number
        ave_n = 0
        good_n = 0
        for i in env.agents:
            if i.adversary:
                ave_n += 1
            else:
                good_n += 1
        print("adversary ", ave_n, "target ", good_n)
        # print("ave_n", ave_n)
        n = env.n
        actors = []
        critics = []
        brains = []
        exploration_noise = []
        observation_dim = []
        action_dim = []
        total_action_dim = 0

        # Aversary Agents action spaces
        for i in range(ave_n):
            total_action_dim = total_action_dim + env.action_space[i].n

        print("total_action_dim", total_action_dim)

        for i in range(n):

            observation_dim.append(env.observation_space[i].shape[0])
            action_dim.append(env.action_space[i].n) # assuming discrete action space here -> otherwise change to something like env.action_space[i].shape[0]
            actors.append(ActorNetwork(sess,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau'])))
            # critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            
            
            if i < ave_n:
                #MADDPG - centralized Critic
                critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            else:
                # DDPG
                critics.append(CriticNetwork(sess,n,observation_dim[i],action_dim[i],float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            
            exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))

        """
        print("Test predict")
        s = env.reset()
        # print(s[0])
        actions = []
        for index in range(len(actors)):
            state_input = np.reshape(s[index],(-1,actors[index].state_dim))
            
            actions.append(actors[index].predict(state_input))

            actors[index].predict_target(state_input)


        actions1 = actions[:ave_n]
        actions2 = actions[ave_n:]
        a_temp1 = np.transpose(np.asarray(actions1),(1,0,2))
        a_for_critic1 = np.asarray([x.flatten() for x in a_temp1])
        a_temp2 = np.transpose(np.asarray(actions2),(1,0,2))
        a_for_critic2 = np.asarray([x.flatten() for x in a_temp2])
        for index in range(len(critics)):
            state_input = np.reshape(s[index],(-1,actors[index].state_dim))
            if index < ave_n:
                critics[index].predict_target(state_input, a_for_critic1)
                #critics[index].predict(state_input, a_for_critic1)
            else:
                critics[index].predict_target(state_input, a_for_critic2)
                #critics[index].predict(state_input, a_for_critic2)
        """ 
        
        # if args['use_gym_monitor']:
        #    if not args['render_env']:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], video_callable=False, force=True)
        #    else:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], force=True)

        # n brains
        if False:
            for i in range(n):
                observation_dim.append(env.observation_space[i].shape[0])
                action_dim.append(env.action_space[i].n)
                brains.apppen(Brain(sess, observation_dim[i], action_dim[i], float(args['actor_lr']), float(args['tau']), \
                                   observation_dim[i], total_action_dim, float(args['critic_lr']), float(args['tau']),float(args['gamma'])))
                exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i]))) 

            # learn()

        if args["runTest"]:

            # , force=True
            # env = wrappers.Monitor(env, args["monitor_dir"], force=True)

            for i in range(n):
                # load model
                actors[i].mainModel.load_weights(args["modelFolder"]+ "ep200000/" +str(i)+'_weights'+'.h5')
                # episode 4754
            import time
            #   time.sleep(3)
            for ep in range(10):
                s = env.reset()
                reward = 0.0
                for step in range(200):
                    
                    time.sleep(0.01)
                    env.render()
                    actions = []
                    for i in range(env.n):
                        state_input = np.reshape(s[i],(-1,env.observation_space[i].shape[0]))
                        noise = OUNoise(mu = np.zeros(5))
                        # predict_action = actors[i].predict(state_input) #+ exploration_noise[i]()
                        # actions.append(predict_action.reshape(env.action_space[i].n,))
                        # +noise()
                        actions.append((actors[i].predict(np.reshape(s[i],(-1, actors[i].mainModel.input_shape[1])))).reshape(actors[i].mainModel.output_shape[1],))
                    #print("{}".format(actions))
                    s, r, d, s2 = env.step(actions)
                    for i in range(env.n):
                        reward += r[i]
                    if np.all(d):
                        break
                print("Episode: {:d}  | Reward: {:f}".format(ep, reward))
            env.close()
            import sys
            sys.exit("test over!")

        if False:
            import time
            # , force=True
            # env = wrappers.Monitor(env, args["monitor_dir"], force=True)
            for ep in range(10):
                # load model
                s = env.reset()
                for j in range(env.n):
                    actors[j].mainModel.load_weights(args["modelFolder"]+ str(j) +'_weights'+'.h5')
                for step in range(300):
                    
                    reward = 0.0
                    # time.sleep(0.05)
                    env.render()
                    actions = []
                    for i in range(env.n):
                        state_input = np.reshape(s[i],(-1,env.observation_space[i].shape[0]))
                        noise = OUNoise(mu = np.zeros(5))
                        # predict_action = actors[i].predict(state_input) #+ exploration_noise[i]()
                        # actions.append(predict_action.reshape(env.action_space[i].n,))
                        # +noise()
                        actions.append((actors[i].predict(np.reshape(s[i],(-1, actors[i].mainModel.input_shape[1])))).reshape(actors[i].mainModel.output_shape[1],))
                    s, r, d, s2 = env.step(actions)
                    for i in range(env.n):
                        reward += r[i]
                    if np.all(d):
                        break
                print("Episode: {:d}  | Reward: {:f}".format(ep, reward))
            
        else:
            if False: 
                train(sess,env,args,actors,critics,exploration_noise, ave_n)
            else:
                global graph, global_queue, update_event, rolling_event, global_step_max, global_step, coord, brain
                graph = tf.get_default_graph()
                global_queue = mp.Queue()
                update_event, rolling_event = mp.Event(), mp.Event()
                global_step_max, global_step = 200*5000, 0
                coord = tf.train.Coordinator()
                brain = Brain(args["modelFolder"])

                distributed_train(sess, env, args, actors, critics, exploration_noise, ave_n)
        #if args['use_gym_monitor']:
        #    envMonitor.monitor.close()
# Training stop
def run():
    env  = make_env.make_env('simple_tag')
    n = env.n
    exploration_noise = []
    actors = []
    for i in range(n):
        # load model
        actors.append(load_model(args["modelFolder"] + str(i) + ".h5"))
        
        exploration_noise.append(OUNoise(mu = np.zeros(env.action_space[i].n)))

    # test for 100 episode
    noise = OUNoise(mu = np.zeros(5))
    import time
    for ep in range(50):
        s = env.reset()
        #if ep == 0:
            #print([i.state.p_pos for i in env.world.borders])
        reward = 0.0
        for step in range(100):
            # time.sleep(0.05)
            env.render()
            actions = []
            for i in range(env.n):
                state_input = np.reshape(s[i],(-1,env.observation_space[i].shape[0]))
                predict_action = actors[i].predict(state_input) #+ noise()
                actions.append(predict_action.reshape(env.action_space[i].n,))
            s, r, d, s2 = env.step(actions)
            for i in range(env.n):
                reward += r[i]
            if np.all(d):
                break

        print("Episode: {:5.2f}  | Reward: {:f}".format(ep, reward))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=10000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/videos/video1')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/2vs1_distributed/tfdata/')
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/2vs1_distributed/weights/")
    parser.add_argument('--runTest', help='use saved model to run', default=False)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    #pp.pprint(args)

    ## Distributed

    main(args)





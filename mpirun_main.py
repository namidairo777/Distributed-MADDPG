# from gym import wrappers
import make_env
import numpy as np
import random
from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcritic_dis import ActorNetwork,CriticNetwork
import argparse
import os
import multiprocessing as mp
import tensorflow as tf
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def prioritized_batch(replay, critics, m_size, n_size):
    """
    1. sample m_size batch from replay memory
    2. calculating td loss
    3. ranking
    4. select n_size td loss with higher td loss
    5. use it for 
    """

    s_batch,a_batch,r_batch,d_batch,s2_batch = replayMemory.miniBatch(m_size)
           
def build_summaries(n):
    """
    Tensorboard summary for losses or rewards
    """
    losses = [tf.Variable(0.) for i in range(n)]
    for i in range(n):
        tf.summary.scalar("Reward_Agent" + str(i), losses[i])
    summary_vars = losses
    summary_ops = tf.summary.merge_all()
    return summary_ops, summary_vars

def saveWeights(actor, i, pathToSave):
    """
    Save model weights
    """
    actor.mainModel.save_weights(pathToSave + str(i) + "_weights.h5")

def showReward(episode_reward, n, ep, start):
    reward_string = ""
    for re in episode_reward:
        reward_string += " {:5.2f} ".format(re)
    print ('|Episode: {:4d} | Time: {:2d} | Rewards: {:s}'.format(ep, int(time.time() - start), reward_string))

def distributed_train_every_step(sess, env, args, actors, critics, noise, ave_n):
    """
    1. replay memory
        - for each timestep
        2. async batch data 
        3. 
    """
    summary_ops,summary_vars = build_summaries(env.n)
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
    replayMemory = ReplayMemory(int(args['buffer_size']),int(args['random_seed']))

    # split_dis = int(int(args['max_episode_len']) / size)
    # batch_index_count = split_dis

    start_time = 0.0
    end_time = 0.0

    for ep in range(int(args['max_episodes'])):
        # collecting reward 
        #batch_index_count = split_dis
        s = env.reset()
        episode_reward = np.zeros((env.n,))
        # weights_data = []
        start = time.time()

        for step in range(int(args['max_episode_len'])):
            action_dims_done = 0
            a = []
            for i in range(env.n):
                actor = actors[i]
                state_input = np.reshape(s[i],(-1, actor.state_dim))
                a.append(actor.act(state_input, noise[i]()).reshape(actor.action_dim,))
            s2, r, done, _ = env.step(a) # a is a list with each element being an array
            #if ep % 10 == 0: 
            #    env.render()
            replayMemory.add(s, a, r, done, s2)
            episode_reward += r
            s = s2
            if replayMemory.size() > int(args["minibatch_size"]):

                # send weights to workers
                critic_weights = [critic.mainModel.get_weights() for critic in critics]
                for i in range(1, size):
                    comm.send(critic_weights, dest=i, tag=9)
                
                # MADDPG Adversary Agent            
                for i in range(ave_n):
                    actor = actors[i]
                    critic = critics[i]
                    if replayMemory.size() > int(args['m_size']):
                        s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args['m_size']))
                        a = []
                        for j in range(ave_n):
                            state_batch_j = np.asarray([x for x in s_batch[:,j]]) #batch processing will be much more efficient even though reshaping will have to be done
                            a.append(actors[j].predict_target(state_batch_j))
                        a_temp = np.transpose(np.asarray(a),(1,0,2))
                        a_for_critic = np.asarray([x.flatten() for x in a_temp])
                        s2_batch_i = np.asarray([x for x in s2_batch[:,i]]) 
                        targetQ = critic.predict_target(s2_batch_i,a_for_critic)
                        yi = []
                        for k in range(int(args['m_size'])):
                            if d_batch[:,i][k]:
                                yi.append(r_batch[:,i][k])
                            else:
                                yi.append(r_batch[:,i][k] + critic.gamma*targetQ[k])
                        # a2 = actor.predict_target(s_batch)                    
                        # Q_target = critic.predict_target(s2_batch, a2)
                        # y = r + gamma * Q_target
                        # TD loss = yi - critic.predict(s_batch, a_batch)               
                        s_batch_i = np.asarray([x for x in s_batch[:,i]])
                        a_batch_data = np.asarray([x.flatten() for x in a_batch[:, 0: ave_n, :]])
                        target_q = np.asarray(yi)
                        #############################################
                        ##   prioritized_batch
                        #############################################
                        # loss = batch
                        losses = []
                        # clip
                        index = 0
                        # number of losses
                        loss_num = int(int(args['m_size']) / int(args['n_size']))


                        # send batch data to workers
                        for i in range(loss_num):
                            data = (s_batch_i[index:index+int(args["n_size"])], 
                                                   a_batch_data[index:index+int(args["n_size"])], 
                                                   target_q[index:index+int(args["n_size"])])
                            comm.send(data, dest=i+1, tag=9)

                            index += int(args["n_size"])
                        
                        # recieve loss from workers
                        for i in range(loss_num):
                            losses.append(comm.recv(source=i+1, tag=9))

                        # which has max loss
                        sorted_index = np.argsort(losses).tolist()
                        max_index = sorted_index[-1]
                        # clip index
                        head = max_index * int(args["n_size"])
                        tail = head +  (args["n_size"])
                        # clipped batch data with higher losses
                        prioritized_a_batch = a_batch_data[head: tail] 
                        prioritized_s_batch = s_batch_i[head: tail] 
                        prioritized_target_q = target_q[head: tail]
                        #############################################
                        ##   prioritized_batch
                        #############################################
                        # critic train
                        critic.train(prioritized_s_batch, prioritized_a_batch, prioritized_target_q)
                        actions_pred = []
                        # for j in range(ave_n):
                        for j in range(ave_n):
                            state_batch_j = np.asarray([x for x in  s2_batch[:,j]])
                            actions_pred.append(actors[j].predict(state_batch_j[head: tail])) 
                        a_temp = np.transpose(np.asarray(actions_pred),(1,0,2))
                        a_for_critic_pred = np.asarray([x.flatten() for x in a_temp])
                        grads = critic.action_gradients(prioritized_s_batch, a_for_critic_pred)[:,action_dims_done:action_dims_done + actor.action_dim]
                        # actor train
                        actor.train(prioritized_s_batch, grads)
                    action_dims_done = action_dims_done + actor.action_dim

            if replayMemory.size() > int(args["minibatch_size"]):
                # Only DDPG agent           
                for i in range(ave_n, env.n):
                    actor = actors[i]
                    critic = critics[i]
                    if replayMemory.size() > int(args["minibatch_size"]):
                        s_batch, a_batch, r_batch, d_batch, s2_batch = replayMemory.miniBatch(int(args["minibatch_size"]))                                  
                        s_batch_i = np.asarray([x for x in s_batch[:,i]])
                        action = np.asarray(actor.predict_target(s_batch_i))
                        action_for_critic = np.asarray([x.flatten() for x in action])
                        s2_batch_i = np.asarray([x for x in s2_batch[:, i]])
                        targetQ = critic.predict_target(s2_batch_i, action_for_critic)
                        y_i = []
                        for k in range(int(args['minibatch_size'])):
                            if d_batch[:, i][k]:
                                y_i.append(r_batch[:, i][k])
                            else:
                                y_i.append(r_batch[:, i][k] + critic.gamma * targetQ[k])
                        s_batch_i= np.asarray([x for x in s_batch[:, i]])
                        critic.train(s_batch_i, np.asarray([x.flatten() for x in a_batch[:, i]]), np.asarray(y_i))
                        action_for_critic_pred = actor.predict(s2_batch_i)
                        gradients = critic.action_gradients(s_batch_i, action_for_critic_pred)[:, :]
                        actor.train(s_batch_i, gradients)                     
                for i in range(0, env.n):
                    actor = actors[i]
                    critic = critics[i]
                    actor.update_target()
                    critic.update_target()


            if step == int(args["max_episode_len"])-1 or np.all(done):
                #############################################
                ##   Record reward data into tensorboard
                #############################################
                ave_reward = 0.0
                good_reward = 0.0
                for i in range(env.n):
                    if i < ave_n:
                        ave_reward += episode_reward[i]
                    else:
                        good_reward += episode_reward[i]
                #summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: episode_reward, summary_vars[1]: episode_av_max_q/float(stp)})
                summary_str = sess.run(summary_ops, feed_dict = {summary_vars[0]: ave_reward, summary_vars[1]: good_reward})
                # summary_str = sess.run(summary_ops, feed_dict = {summary_vars[i]: losses[i] for i in range(len(losses))})
                writer.add_summary(summary_str, ep)
                writer.flush()
                showReward(episode_reward, env.n, ep, start)
                break

        if ep % 50 == 0 and ep != 0:
            print("Starting saving model weights every 50 episodes")
            for i in range(env.n):
                saveWeights(actors[i], i, args["modelFolder"])
            print("Model weights saved")
        if ep % 100 == 0 and ep != 0:
            directory = args["modelFolder"] + "ep" + str(ep) + "/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            print("Starting saving model weights to folder every 100 episodes")
            for i in range(env.n):
                saveWeights(actors[i], i, directory)
            print("Model weights saved to folder")            



def collect_batch(env, args, critics, ave_n):
    
    for ep in range(int(args['max_episodes']) * int(args['max_episode_len'])):
        # recieve weights
        weights = comm.recv(source=0, tag=9)
     
        # set weights
        for i in range(len(critics)):
            critics[i].mainModel.set_weights(weights[i])

        # receieve batch data for every predator agent to calculate loss
        for i in range(ave_n):
            # recieve data from i agent 
            (s_batch, a_batch, target_q) = comm.recv(source=0, tag=9)
            loss = critics[i].get_loss(s_batch, a_batch, target_q)
            # send loss
            comm.send(loss, dest=0, tag=9)

def main(args):
    # Master
    if rank == 0:       
        #######################
        # Setting up:
        # - environment, random seed
        # - tensorflow option
        # - network
        # - replay
        #########################
        if not os.path.exists(args["modelFolder"]):
            os.makedirs(args["modelFolder"])
        if not os.path.exists(args["summary_dir"]):
            os.makedirs(args["summary_dir"])
        # env and random seed
        env = make_env.make_env('simple_tag')
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        # tensorflow
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # agent number
            n = env.n
            ave_n = 0
            good_n = 0
            for i in env.agents:
                if i.adversary:
                    ave_n += 1
                else:
                    good_n += 1
            # Actor Critic
            n = env.n
            actors = []
            critics = []
            exploration_noise = []
            observation_dim = []
            action_dim = []
            total_action_dim = 0

            # Aversary Agents action spaces
            for i in range(ave_n):
                total_action_dim = total_action_dim + env.action_space[i].n
            # print("total_action_dim {} for cooperative agents".format(total_action_dim))
            for i in range(n):
                observation_dim.append(env.observation_space[i].shape[0])
                action_dim.append(env.action_space[i].n) # assuming discrete action space here -> otherwise change to something like env.action_space[i].shape[0]
                actors.append(ActorNetwork(sess,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau'])))
                # critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
                if i < ave_n:
                    # MADDPG - centralized Critic
                    critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
                else:
                    # DDPG
                    critics.append(CriticNetwork(sess,n,observation_dim[i],action_dim[i],float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            
                exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))

            distributed_train_every_step(sess, env, args, actors, critics, exploration_noise, ave_n)    
    # Worker
    else:
        #######################
        # Setting up:
        # - tensorflow option
        # - network
        # 
        #
        env = make_env.make_env('simple_tag')
        np.random.seed(int(args['random_seed']) + rank)
        tf.set_random_seed(int(args['random_seed']) + rank)
        env.seed(int(args['random_seed']) + rank) 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.08)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            # agent number
            n = env.n
            ave_n = 0
            good_n = 0
            for i in env.agents:
                if i.adversary:
                    ave_n += 1
                else:
                    good_n += 1
            # Actor Critic
            n = env.n
            critics = []
            observation_dim = []
            total_action_dim = 0
            # Aversary Agents action spaces
            for i in range(ave_n):
                total_action_dim = total_action_dim + env.action_space[i].n
            for i in range(ave_n):
                observation_dim.append(env.observation_space[i].shape[0])
                critics.append(CriticNetwork(sess, n, observation_dim[i], total_action_dim, float(args['critic_lr']), float(args['tau']), float(args['gamma'])))
 
            collect_batch(env, args, critics, ave_n)

if __name__ == '__main__':

    print("Start to work!")
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='provide arguments for Distributed-MADDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--prioritized-alpha', help='prioritized alpha', default=0.6)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=128)

    # run parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=10000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/videos/video1')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/2vs1_dis_prioritizedBatch/tfdata_critic_worker/')
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/2vs1_dis_prioritizedBatch/weights_critic_worker/")
    parser.add_argument('--runTest', help='use saved model to run', default=False)
    parser.add_argument('--work-max-step', help='work_max_step', default=1)
    parser.add_argument('--m-size', help='M size', default=128)
    parser.add_argument('--n-size', help='N size', default=64)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    main(args)





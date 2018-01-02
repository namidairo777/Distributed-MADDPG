# from gym import wrappers
import make_env
import numpy as np
import random
from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcritic_dis import ActorNetwork,CriticNetwork
# from Distributed_Train import distributed_train
import argparse
# from keras.models import load_model
import os
import multiprocessing as mp
import tensorflow as tf
import time


def test(args):
    # env and random seed
    env = make_env.make_env('simple_tag')
    np.random.seed(int(args['random_seed'])+100)
    tf.set_random_seed(int(args['random_seed']))
    # env.seed(int(args['random_seed']))
    # tensorflow
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    # config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
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

        for i in range(ave_n):
            total_action_dim = total_action_dim + env.action_space[i].n
        for i in range(n):
            observation_dim.append(env.observation_space[i].shape[0])
            action_dim.append(env.action_space[i].n) # assuming discrete action space here -> otherwise change to something like env.action_space[i].shape[0]
            actors.append(ActorNetwork(sess,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau'])))
            if i < ave_n:
                # MADDPG - centralized Critic
                critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            else:
                # DDPG
                critics.append(CriticNetwork(sess,n,observation_dim[i],action_dim[i],float(args['critic_lr']),float(args['tau']),float(args['gamma'])))        
            exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))
        for i in range(n):
            actors[i].mainModel.load_weights(args["modelFolder"] + str(i)+'_weights'+'.h5')   #"ep200/" + 
        # s = env.reset()
        # s = env.reset()
        ave_reward = 0.0
        for ep in range(100):
            s = env.reset()
            reward = 0.0
            for step in range(200):
                env.render()
                # time.sleep(10)
                actions = []
                for i in range(env.n):
                    state_input = np.reshape(s[i],(-1,env.observation_space[i].shape[0]))
                    noise = OUNoise(mu = np.zeros(5))                    
                    actions.append((actors[i].predict(np.reshape(s[i],(-1, actors[i].mainModel.input_shape[1])))).reshape(actors[i].mainModel.output_shape[1],))
                s, r, d, s2 = env.step(actions)
                for i in range(env.n):
                    reward += r[i]
                if np.all(d):
                    break
            print("Episode: {:d}  | Reward: {:f}".format(ep, reward))
            ave_reward += reward
        env.close()
        print("Average reward: {} of 10 episodes".format(ave_reward/100))
        import sys
        sys.exit("test over!")


if __name__ == '__main__':
    print("Start to work")
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
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/2vs1_dis_prioritizedBatch/tfdata/')
    # comparison_of_3/tensorboard_data/weights/proposed_weights/
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/2vs1_dis_prioritizedBatch/weights_critic_worker_256sample/") #2vs1_dis_prioritizedBatch/weights_critic_worker/ 2vs1_maddpg_tanh/weights_prioritized/
    parser.add_argument('--runTest', help='use saved model to run', default=False)
    parser.add_argument('--work-max-step', help='work_max_step', default=50)
    parser.add_argument('--m-size', help='M size', default=128)
    parser.add_argument('--n-size', help='N size', default=64)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    test(args)
    # main(args)





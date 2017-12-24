import make_env
import numpy as np
import random
from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcritic_dis import ActorNetwork,CriticNetwork
from Train_maddpg_prioritized import train
import argparse
from keras.models import load_model
import os
import tensorflow as tf


def main(args):
    if not os.path.exists(args["modelFolder"]):
        os.makedirs(args["modelFolder"])
    if not os.path.exists(args["summary_dir"]):
        os.makedirs(args["summary_dir"])

    #with tf.device("/gpu:0"):
    # MADDPG for Ave Agent
    # DDPG for Good Agent
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    config = tf.ConfigProto(
        device_count = {'CPU': 0}
    )
    # config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        env  = make_env.make_env('simple_tag')

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))
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
                # MADDPG - centralized Critic
                critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            else:
                # DDPG
                critics.append(CriticNetwork(sess,n,observation_dim[i],action_dim[i],float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
            
            exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))

        train(sess,env,args,actors,critics,exploration_noise, ave_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=128)

    # run parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=10000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/videos/video1')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/4vs2_maddpg_tanh/tfdata/')
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/4vs2_maddpg_tanh/weights/")
    parser.add_argument('--runTest', help='use saved model to run', default=False)
    parser.add_argument('--m-size', help='M size', default=128)
    parser.add_argument('--n-size', help='N size', default=64)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    main(args)





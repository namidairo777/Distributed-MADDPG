import tensorflow as tf
from gym import wrappers
import make_env
import numpy as np
#import random
#from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcriticv2 import ActorNetwork,CriticNetwork
from Train import train
import argparse

def main(args):

    with tf.Session() as sess:
        env  = make_env.make_env('simple_tag')
        n = env.n
        actors = []
        critics = []
        exploration_noise = []
        observation_dim = []
        action_dim = []
        total_action_dim = 0
        for i in range(n):
            total_action_dim = total_action_dim + env.action_space[i].n
        for i in range(n):
            observation_dim.append(env.observation_space[i].shape[0])
            action_dim.append(env.action_space[i].n) # assuming discrete action space here -> otherwise change to something like env.action_space[i].shape[0]
            actors.append(ActorNetwork(sess,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau'])))
            critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['actor_lr']),float(args['tau']),float(args['gamma'])))
            exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))

        #if args['use_gym_monitor']:
        #    if not args['render_env']:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], video_callable=False, force=True)
        #    else:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess,env,args,actors,critics,exploration_noise)
        #if args['use_gym_monitor']:
        #    envMonitor.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.01)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.01)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg_3')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg_3')

    parser.set_defaults(render_env=True)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    #pp.pprint(args)

    main(args)
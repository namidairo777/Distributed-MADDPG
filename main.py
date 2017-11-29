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
from keras.models import load_model

def main(args):

    #with tf.device("/gpu:0"):
    # MADDPG for Ave Agent
    # DDPG for Good Agent
    with tf.Session() as sess:
        if args["runTest"]:
            run()
            import sys
            sys.exit("test over!")

        env  = make_env.make_env('simple_tag')

        # Calculate good and ave agents number
        ave_n = 0
        good_n = 0
        for i in env.agents:
            if i.adversary:
                ave_n += 1
            else:
                good_n += 1

        # print("ave_n", ave_n)
        n = env.n
        actors = []
        critics = []
        exploration_noise = []
        observation_dim = []
        action_dim = []
        total_action_dim = 0

        # Aversary Agents action spaces
        for i in range(env.n):
            total_action_dim = total_action_dim + env.action_space[i].n

        # print("total_action_dim", total_action_dim)

        for i in range(n):

            observation_dim.append(env.observation_space[i].shape[0])
            action_dim.append(env.action_space[i].n) # assuming discrete action space here -> otherwise change to something like env.action_space[i].shape[0]
            actors.append(ActorNetwork(sess,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau'])))
            
            # if i < ave_n:
                # MADDPG - centralized Critic
            critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['actor_lr']),float(args['tau']),float(args['gamma'])))
            # else:
                # DDPG
                # critics.append(CriticNetwork(sess,n,observation_dim[i],action_dim[i],float(args['actor_lr']),float(args['tau']),float(args['gamma'])))
            
            exploration_noise.append(OUNoise(mu = np.zeros(action_dim[i])))


        #if args['use_gym_monitor']:
        #    if not args['render_env']:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], video_callable=False, force=True)
        #    else:
        #        envMonitor = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess,env,args,actors,critics,exploration_noise, ave_n)
        #if args['use_gym_monitor']:
        #    envMonitor.monitor.close()

def main_DMADDPG(args):

    with tf.Session() as sess:

        env = make_env.make_env("simple_tag")

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
    import time
    for ep in range(50):
        s = env.reset()
        #if ep == 0:
            #print([i.state.p_pos for i in env.world.borders])
        reward = 0.0
        for step in range(150):
            time.sleep(0.05)
            env.render()
            actions = []
            for i in range(env.n):
                state_input = np.reshape(s[i],(-1,env.observation_space[i].shape[0]))
                predict_action = actors[i].predict(state_input) #+ exploration_noise[i]()
                actions.append(predict_action.reshape(env.action_space[i].n,))
            s, r, d, s2 = env.step(actions)
            for i in range(env.n - 1):
                reward += r[i]
            if np.all(d):
                break

        print("Episode: {:d}  | Reward: {:f}".format(ep, reward))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.01)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.01)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.95)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    #parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='MountainCarContinuous-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=5000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=500)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg_4')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/maddpg2_vs_ddpg_1/tf_data')
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/maddpg2_vs_ddpg_1/keras_model/actor_maddpg_")
    parser.add_argument('--runTest', help='use saved model to run', default=False)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    #pp.pprint(args)

    main(args)

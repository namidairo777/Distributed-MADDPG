from gym import wrappers
import make_env
import numpy as np
import random
from ReplayMemory import ReplayMemory
from ExplorationNoise import OrnsteinUhlenbeckActionNoise as OUNoise
from actorcriticv2 import ActorNetwork,CriticNetwork
from Train_ddpg import train
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
    # config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as sess:
    # with tf.Session(config=config) as sess:

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
            
            
            #if i < ave_n:
                #MADDPG - centralized Critic
                #critics.append(CriticNetwork(sess,n,observation_dim[i],total_action_dim,float(args['critic_lr']),float(args['tau']),float(args['gamma'])))
           # else:
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
                actors[i].mainModel.load_weights(args["modelFolder"] +str(i)+'_weights'+'.h5')
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
            if True: 
                train(sess,env,args,actors,critics,exploration_noise, ave_n)
            else:
                distributed_train(sess, env, args, actors, critics, exploration_noise, ave_n)
        #if args['use_gym_monitor']:
        #    envMonitor.monitor.close()


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
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=5000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/videos/video1')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/2vs1_ddpg/tfdata/')
    parser.add_argument('--modelFolder', help='the folder which saved model data', default="./results/2vs1_ddpg/weights/")
    parser.add_argument('--runTest', help='use saved model to run', default=False)

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    #pp.pprint(args)

    ## Distributed

    main(args)





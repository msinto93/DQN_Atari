'''
## Test ##
# Test a trained DQN. This can be run alongside training by running 'run_every_new_ckpt.sh'.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import scipy.stats as ss

from train import get_train_args
from utils.utils import preprocess_image, reset_env_and_state_buffer
from utils.state_buffer import StateBuffer
from utils.network import DeepQNetwork
    
def get_test_args(train_args):
    test_params = argparse.ArgumentParser()
    
    # Environment parameters (First 4 params must be same as those used in training)
    test_params.add_argument("--env", type=str, default=train_args.env, help="Environment to use (must have RGB image state space and discrete action space)")
    test_params.add_argument("--frame_width", type=int, default=train_args.frame_width, help="Frame width after resize.")
    test_params.add_argument("--frame_height", type=int, default=train_args.frame_height, help="Frame height after resize.")
    test_params.add_argument("--frames_per_state", type=int, default=train_args.frames_per_state, help="Sequence of frames which constitutes a single state.")
    test_params.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment on the screen during testing")
    test_params.add_argument("--random_seed", type=int, default=4321, help="Random seed for reproducability")
    
    # Testing parameters
    test_params.add_argument("--num_eps_test", type=int, default=20, help="Number of episodes to test for")
    test_params.add_argument("--max_ep_length", type=int, default=2000, help="Maximum number of steps per episode")
    test_params.add_argument("--max_initial_random_steps", type=int, default=10, help="Maximum number of random steps to take at start of episode to ensure random starting point")

    # Files/directories
    test_params.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    test_params.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load (if None, load latest ckpt)")
    test_params.add_argument("--log_dir", type=str, default='./logs/test', help="Directory for saving logs")
    test_params.add_argument("--results_dir", type=str, default='./test_results', help="Directory for saving txt file of results")
    test_params.add_argument("--results_file", type=str, default='results.txt', help="Text file of test results (if None, do not save results)")
    
    return test_params.parse_args()

    
def test(args):
    # Create environment
    env = gym.make(args.env)
    num_actions = env.action_space.n
    
    # Set random seeds for reproducability
    env.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    
    # Initialise state buffer
    state_buf = StateBuffer(args)
    
    # Define input placeholders    
    state_ph = tf.placeholder(tf.uint8, (None, args.frame_height, args.frame_width, args.frames_per_state))
    
    # Instantiate DQN network
    DQN = DeepQNetwork(num_actions, state_ph, scope='DQN_main')
    DQN_predict_op = DQN.predict()
        
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)       
                
    # Load ckpt file
    loader = tf.train.Saver()    
    if args.ckpt_file is not None:
        ckpt = args.ckpt_dir + '/' + args.ckpt_file  
    else:
        ckpt = tf.train.latest_checkpoint(args.ckpt_dir)
     
    loader.restore(sess, ckpt)
    sys.stdout.write('%s restored.\n\n' % ckpt)
    sys.stdout.flush() 
     
    ckpt_split = ckpt.split('-')
    train_ep = ckpt_split[-1]  
    
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    
    # Create summary op to save episode reward to Tensorboard log
    reward_var = tf.Variable(0.0, trainable=False)
    tf.summary.scalar("Average Test Reward", reward_var)
    summary_op = tf.summary.merge_all()
    
        
    ## Begin testing
                       
    env.reset()
    rewards = []   
    
    for test_ep in range(args.num_eps_test):
        # Reset environment and state buffer for next episode
        reset_env_and_state_buffer(env, state_buf, args)  
        ep_reward = 0
        step = 0
        ep_done = False
        
        initial_steps = np.random.randint(1, args.max_initial_random_steps+1)
        
        sys.stdout.write('\n')   
        sys.stdout.flush()
        
        while not ep_done:
            if args.render:
                env.render()
            else:
                env.render(mode='rgb_array')
            
            #Choose random action for initial steps to ensure every episode has a random start point. Then choose action with highest Q-value according to network's current policy.
            if step < initial_steps:
                test_action = env.action_space.sample()
            else:
                test_state = np.expand_dims(state_buf.get_state(), 0)
                test_action = sess.run(DQN_predict_op, {state_ph:test_state})
            
            test_frame, test_reward, test_ep_terminal, _ = env.step(test_action)
            
            test_frame = preprocess_image(test_frame, args.frame_width, args.frame_height)
            state_buf.add(test_frame)    
            
            ep_reward += test_reward
            step += 1
            
            sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d} \t Steps = {:d} \t Reward = {:.2f}'.format(test_ep, args.num_eps_test, step, ep_reward))
            sys.stdout.flush() 
      
            # Episode can finish either by reaching terminal state or max episode steps
            if test_ep_terminal or step == args.max_ep_length:
                rewards.append(ep_reward)
                ep_done = True   
            
    mean_reward = np.mean(rewards)
    error_reward = ss.sem(rewards)
            
    sys.stdout.write('\n\nTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
    sys.stdout.flush()  
            
    # Log average episode reward for Tensorboard visualisation
    summary_str = sess.run(summary_op, {reward_var: mean_reward})
    summary_writer.add_summary(summary_str, train_ep)
     
    # Write results to file        
    if args.results_file is not None:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        output_file = open(args.results_dir + '/' + args.results_file, 'a')
        output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(train_ep, mean_reward, error_reward))
        output_file.flush()
        sys.stdout.write('Results saved to file \n\n')
        sys.stdout.flush()      
    
    env.close()          
                  

if  __name__ == '__main__':
    train_args = get_train_args()
    test_args = get_test_args(train_args)
    test(test_args)         
            
        
    
    
        
        
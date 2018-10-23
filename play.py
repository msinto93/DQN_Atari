'''
## Play ##
# Run a trained DQN on an Open AI gym environment and observe its performance on screen
@author: Mark Sinton (msinto93@gmail.com) 
'''

import time
import argparse
import gym
import tensorflow as tf
import numpy as np

from train import get_train_args
from utils.utils import preprocess_image, reset_env_and_state_buffer
from utils.state_buffer import StateBuffer
from utils.network import DeepQNetwork
    
def get_play_args(train_args):
    parser = argparse.ArgumentParser()
    
    # Environment parameters (First 4 params must be same as those used in training)
    parser.add_argument("--env", type=str, default=train_args.env, help="Environment to use (must have RGB image state space and discrete action space)")
    parser.add_argument("--frame_width", type=int, default=train_args.frame_width, help="Frame width after resize.")
    parser.add_argument("--frame_height", type=int, default=train_args.frame_height, help="Frame height after resize.")
    parser.add_argument("--frames_per_state", type=int, default=train_args.frames_per_state, help="Sequence of frames which constitutes a single state.")
    parser.add_argument("--random_seed", type=int, default=4321, help="Random seed for reproducability")
    
    # Play parameters
    parser.add_argument("--num_eps", type=int, default=10, help="Number of episodes to run for")
    parser.add_argument("--max_ep_length", type=int, default=2000, help="Maximum number of steps per episode")
    parser.add_argument("--max_initial_random_steps", type=int, default=10, help="Maximum number of random steps to take at start of episode to ensure random starting point")
    
    # Files/directories
    parser.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for loading checkpoints")
    parser.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load (if None, load latest ckpt)")
    
    return parser.parse_args()

    
def play(args):
    # Create environment
    env = gym.make(args.env)
    num_actions = env.action_space.n
    
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
    print('%s restored.\n\n' % ckpt)
       
    for ep in range(0, args.num_eps):
        # Reset environment and state buffer for next episode
        reset_env_and_state_buffer(env, state_buf, args)  
        step = 0
        ep_done = False
        initial_steps = np.random.randint(1, args.max_initial_random_steps+1)
        
        while not ep_done:
            time.sleep(0.05)
            env.render()

            #Choose random action for initial steps to ensure every episode has a random start point. Then choose action with highest Q-value according to network's current policy.
            if step < initial_steps:
                action = env.action_space.sample()
            else:
                state = np.expand_dims(state_buf.get_state(), 0)
                action = sess.run(DQN_predict_op, {state_ph:state})
            
            frame, _, ep_terminal, _ = env.step(action)
            frame = preprocess_image(frame, args.frame_width, args.frame_height)
            state_buf.add(frame)    
            step += 1
            
            # Episode can finish either by reaching terminal state or max episode steps
            if ep_terminal or step == args.max_ep_length:
                ep_done = True   
  
                         
                  

if  __name__ == '__main__':
    train_args = get_train_args()
    play_args = get_play_args(train_args)
    play(play_args)         
            
        
    
    
        
        
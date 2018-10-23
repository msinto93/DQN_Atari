'''
## Train ##
# Code to train Deep Q Network on OpenAI Gym environments
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import time
import random

from utils.utils import preprocess_image, reset_env_and_state_buffer
from utils.experience_replay import ReplayMemory   
from utils.state_buffer import StateBuffer
from utils.network import DeepQNetwork
    
def get_train_args():
    train_params = argparse.ArgumentParser()
    
    # Environment parameters
    train_params.add_argument("--env", type=str, default='BreakoutDeterministic-v4', help="Environment to use (must have RGB image state space and discrete action space)")
    train_params.add_argument("--render", type=bool, default=False, help="Whether or not to display the environment on the screen during training")
    train_params.add_argument("--random_seed", type=int, default=1234, help="Random seed for reproducability")
    train_params.add_argument("--frame_width", type=int, default=105, help="Frame width after resize.")
    train_params.add_argument("--frame_height", type=int, default=80, help="Frame height after resize.")
    train_params.add_argument("--frames_per_state", type=int, default=4, help="Sequence of frames which constitutes a single state.")
    
    # Training parameters
    train_params.add_argument("--num_steps_train", type=int, default=50000000, help="Number of steps to train for")    
    train_params.add_argument("--train_frequency", type=int, default=4, help="Perform training step every N game steps.")
    train_params.add_argument("--max_ep_steps", type=int, default=2000, help="Maximum number of steps per episode")
    train_params.add_argument("--batch_size", type=int, default=32)
    train_params.add_argument("--learning_rate", type=float, default=0.00025)
    train_params.add_argument("--replay_mem_size", type=int, default=1000000, help="Maximum size of replay memory buffer")
    train_params.add_argument("--initial_replay_mem_size", type=int, default=50000, help="Initial size of replay memory (populated by random actions) before learning can start")
    train_params.add_argument("--epsilon_start", type=float, default=1.0, help="Exploration rate at the beginning of training.")
    train_params.add_argument("--epsilon_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
    train_params.add_argument("--epsilon_step_end", type=int, default=1000000, help="After how many steps to stop decaying the exploration rate.")
    train_params.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate (gamma) for future rewards.")
    train_params.add_argument("--update_target_step", type=float, default=10000, help="Copy current network parameters to target network every N steps.")
    train_params.add_argument("--save_ckpt_step", type=float, default=250000, help="Save checkpoint every N steps")
    train_params.add_argument("--save_log_step", type=int, default=1000, help="Save logs every N steps")
    
    # Files/directories
    train_params.add_argument("--ckpt_dir", type=str, default='./ckpts', help="Directory for saving/loading checkpoints")
    train_params.add_argument("--ckpt_file", type=str, default=None, help="Checkpoint file to load and resume training from (if None, train from scratch)")
    train_params.add_argument("--log_dir", type=str, default='./logs/train', help="Directory for saving logs")
    
    return train_params.parse_args()
           
    
def train(args):
    
    # Function to return exploration rate based on current step
    def exploration_rate(current_step, exp_rate_start, exp_rate_end, exp_step_end):
        if current_step < exp_step_end:
            exploration_rate = current_step * ((exp_rate_end-exp_rate_start)/(float(exp_step_end))) + 1
        else:
            exploration_rate = exp_rate_end
            
        return exploration_rate
    
    # Function to update target network parameters with main network parameters
    def update_target_network(from_scope, to_scope):    
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)    
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
    
        op_holder = []
        
        # Update old network parameters with new network parameters
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))
        
        return op_holder
    
    
    # Create environment
    env = gym.make(args.env)
    num_actions = env.action_space.n
    
    # Initialise replay memory and state buffer
    replay_mem = ReplayMemory(args)
    state_buf = StateBuffer(args)
    
    # Define input placeholders    
    state_ph = tf.placeholder(tf.uint8, (None, args.frame_height, args.frame_width, args.frames_per_state))
    action_ph = tf.placeholder(tf.int32, (None))
    target_ph = tf.placeholder(tf.float32, (None))
    
    # Instantiate DQN network
    DQN = DeepQNetwork(num_actions, state_ph, action_ph, target_ph, args.learning_rate, scope='DQN_main')   # Note: One scope cannot be the prefix of another scope (e.g. cannot name this scope 'DQN' and 
                                                                                                            # target network scope 'DQN_target', as a search for vars in 'DQN' scope will return both networks' vars)
    DQN_predict_op = DQN.predict()
    DQN_train_step_op = DQN.train_step()
    
    # Instantiate DQN target network
    DQN_target = DeepQNetwork(num_actions, state_ph, scope='DQN_target')
    
    update_target_op = update_target_network('DQN_main', 'DQN_target')
        
    # Create session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)       
        
    # Add summaries for Tensorboard visualisation
    tf.summary.scalar('Loss', DQN.loss)  
    reward_var = tf.Variable(0.0, trainable=False)
    tf.summary.scalar("Episode Reward", reward_var)
    epsilon_var = tf.Variable(args.epsilon_start, trainable=False)
    tf.summary.scalar("Exploration Rate", epsilon_var)
    summary_op = tf.summary.merge_all() 
        
    # Define saver for saving model ckpts
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(args.ckpt_dir, model_name)        
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    saver = tf.train.Saver(max_to_keep=201)  
    
    # Create summary writer to write summaries to disk
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    
    # Load ckpt file if given
    if args.ckpt_file is not None:
        loader = tf.train.Saver()   #Restore all variables from ckpt
        ckpt = args.ckpt_dir + '/' + args.ckpt_file
        ckpt_split = ckpt.split('-')
        step_str = ckpt_split[-1]
        start_step = int(step_str)    
        loader.restore(sess, ckpt)
    else:
        start_step = 0
        sess.run(tf.global_variables_initializer())
        sess.run(update_target_op)

        
    ## Begin training
                       
    env.reset()
    
    ep_steps = 0
    episode_reward = 0
    episode_rewards = []
    duration_values = []

    # Initially populate replay memory by taking random actions 
    sys.stdout.write('\nPopulating replay memory with random actions...\n')   
    sys.stdout.flush()          
    
    for random_step in range(1, args.initial_replay_mem_size+1):
        
        if args.render:
            env.render()
        else:
            env.render(mode='rgb_array')
        
        action = env.action_space.sample()
        frame, reward, terminal, _ = env.step(action)
        frame = preprocess_image(frame, args.frame_width, args.frame_height)
        replay_mem.add(action, reward, frame, terminal)
        
        if terminal:
            env.reset()
                        
        sys.stdout.write('\x1b[2K\rStep {:d}/{:d}'.format(random_step, args.initial_replay_mem_size))
        sys.stdout.flush() 
    
    # Begin training process         
    reset_env_and_state_buffer(env, state_buf, args)
    sys.stdout.write('\n\nTraining...\n\n')   
    sys.stdout.flush()
    
    for train_step in range(start_step+1, args.num_steps_train+1):      
        start_time = time.time()  
        # Run 'train_frequency' iterations in the game for every training step       
        for _ in range(0, args.train_frequency):
            ep_steps += 1
            
            if args.render:
                env.render()
            else:
                env.render(mode='rgb_array')
            
            # Use an epsilon-greedy policy to select action
            epsilon = exploration_rate(train_step, args.epsilon_start, args.epsilon_end, args.epsilon_step_end)
            if random.random() < epsilon:
                #Choose random action
                action = env.action_space.sample()
            else:
                #Choose action with highest Q-value according to network's current policy
                current_state = np.expand_dims(state_buf.get_state(), 0)
                action = sess.run(DQN_predict_op, {state_ph:current_state})
                   
            # Take action and store experience
            frame, reward, terminal, _ = env.step(action)
            frame = preprocess_image(frame, args.frame_width, args.frame_height)
            state_buf.add(frame)
            replay_mem.add(action, reward, frame, terminal) 
            episode_reward += reward     
            
            if terminal or ep_steps == args.max_ep_steps:  
                # Collect total reward of episode              
                episode_rewards.append(episode_reward)
                # Reset episode reward and episode steps counters
                episode_reward = 0
                ep_steps = 0
                # Reset environment and state buffer for next episode
                reset_env_and_state_buffer(env, state_buf, args)                
        
        ## Training step    
        # Get minibatch from replay mem
        states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch()
        # Calculate target by passing next states through the target network and finding max future Q
        future_Q = sess.run(DQN_target.output, {state_ph:next_states_batch})
        max_future_Q = np.max(future_Q, axis=1)
        # Q values of the terminal states is 0 by definition
        max_future_Q[terminals_batch] = 0
        targets = rewards_batch + (max_future_Q*args.discount_rate)
        
        # Execute training step
        if train_step % args.save_log_step == 0:
            # Train and save logs
            average_reward = sum(episode_rewards)/len(episode_rewards)
            summary_str, _ = sess.run([summary_op, DQN_train_step_op], {state_ph:states_batch, action_ph:actions_batch, target_ph:targets, reward_var: average_reward, epsilon_var: epsilon})
            summary_writer.add_summary(summary_str, train_step)
            # Reset rewards buffer
            episode_rewards = []
        else:
            # Just train
            _ = sess.run(DQN_train_step_op, {state_ph:states_batch, action_ph:actions_batch, target_ph:targets})
        
        # Update target networks    
        if train_step % args.update_target_step == 0:
            sess.run(update_target_op)
        
        # Calculate time per step and display progress to console   
        duration = time.time() - start_time
        duration_values.append(duration)
        ave_duration = sum(duration_values)/float(len(duration_values))
        
        sys.stdout.write('\x1b[2K\rStep {:d}/{:d} \t ({:.3f} s/step)'.format(train_step, args.num_steps_train, ave_duration))
        sys.stdout.flush()       
        
        # Save checkpoint       
        if train_step % args.save_ckpt_step == 0:
            saver.save(sess, checkpoint_path, global_step=train_step)
            sys.stdout.write('\n Checkpoint saved\n')   
            sys.stdout.flush() 
            
            # Reset time calculation
            duration_values = []
            
                  

if  __name__ == '__main__':
    train_args = get_train_args()
    train(train_args)         
            
        
    
    
        
        
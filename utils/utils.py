'''
## Utils ##
@author: Mark Sinton (msinto93@gmail.com) 
'''

import cv2

def to_greyscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def downsample(img, target_width, target_height):
    return cv2.resize(img, (target_width, target_height))

def preprocess_image(img, target_width, target_height):
    # Convert RGB to BGR for cv2
    img = img[:,:,::-1]
    return to_greyscale(downsample(img, target_width, target_height))

def reset_env_and_state_buffer(env, state_buffer, args):
    # Reset the environment (required each time we reach a terminal state)
    # Each time we reset the environment we must add the start frame to the state buffer 'frames_per_state' times to fill up the buffer and give us our first state.
    frame = env.reset()
    state_buffer.reset()
    
    frame = preprocess_image(frame, args.frame_width, args.frame_height)
    
    for _ in range(args.frames_per_state):
        state_buffer.add(frame)    
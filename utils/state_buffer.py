'''
## State buffer ##
# A state is made up of multiple frames. StateBuffer maintains the last 'frames_per_state' frames in a buffer to create a state from.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import numpy as np

class StateBuffer:
    def __init__(self, args):
        self.frames_per_state = args.frames_per_state
        self.dims = (args.frame_height, args.frame_width)
        self.buffer = np.zeros((args.frame_height, args.frame_width, self.frames_per_state), dtype = np.uint8)

    def add(self, frame):
        assert frame.shape == self.dims
        self.buffer[..., :-1] = self.buffer[..., 1:]
        self.buffer[..., -1] = frame
        
    def reset(self):
        self.buffer *= 0
        
    def get_state(self):
        return self.buffer
    

if  __name__ == '__main__':
    ### For testing ###
    import argparse
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_width", type=int, default=3, help="Frame width after resize.")
    parser.add_argument("--frame_height", type=int, default=2, help="Frame height after resize.")
    parser.add_argument("--frames_per_state", type=int, default=4, help="Sequence of frames which constitutes a single state.")
    args = parser.parse_args()
    
    buf = StateBuffer(args)
    
    #Populate experience buffer
    
    for i in range(0,9):
        frame = np.random.randint(255, size=(args.frame_height, args.frame_width))
        action = np.random.randint(4)
        reward = np.random.randint(2)
        terminal = np.random.choice(a=[False, False, False, False, False, False, False, False, True])
    
        buf.add(frame)
        
        state = buf.get_state()
        print(state)
        
        if i == 5:
            buf.reset()

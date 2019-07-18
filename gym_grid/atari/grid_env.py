import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep

from  grid_graphics import *

# Action Codes: 0,1,2,3 : Up, Right, Left and Down respectively
# Stimulation code: [0, 0, 0, agent_x, agent_y, ghost_x, ghost_y, action]
# (1,1) is the starting position, (8,8) is the end position


class ALEInterface(object):
    def __init__(self):
      self._lives_left = 0

    def lives(self):
      return 0 #self.lives_left

class GridEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}

    def __init__(self, max_timesteps=100, max_dist=3, grid_size=(10, 10), tcp_tagging=False, tcp_port=15361):

        # Atari-platform related parameters
        self._atari_dims = (210,160,3)		# Specifies standard atari resolution
        (self._atari_height, self._atari_width, self._atari_channels) = self._atari_dims

        #  Game-related paramteres
        self._screen_height = grid_size[0]
        self._screen_width = grid_size[1]
        self._screen_dims = [self._screen_height, self._screen_width]


        self._actions = [[-1,0],[0,1],[0,-1],[1,0]]     # Up, Right, Left, DOWN
        self._score = 0.0
        self._state = [-1, -1, -1, -1, -1]   # [agent_x, agent_y, target_x, target_y, action]
        self._time = 0
        self._max_timesteps = max_timesteps
        self._max_dist = 3

        # Gym-related variables [must be defined]
        self.action_set = np.array([0,1,2,3],dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self._atari_height, self._atari_width, 3), dtype=np.uint8)
        self.viewer = None

        self._offset = 25



        # Code for TCP Tagging
        self._tcp_tagging = tcp_tagging
        if (self._tcp_tagging):
            self._host = '127.0.0.1'
            self._port = tcp_port
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._s.connect((self._host, self._port))

        # Methods
        self._ale = ALEInterface()
        self.seed()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        if isinstance(action, np.ndarray):
          action = action[0]
        assert self.action_space.contains(action)   # makes sure the action is valid

        self._time = self._time + 1


        # Updating the state, state is hidden from observation
        # Ghost position is the agent old position
        [agent_x, agent_y, target_x, target_y, prev_action] = self._state
        agent_prev_x = agent_x
        agent_prev_y = agent_y
        target_prev_x = target_x
        target_prev_y = target_y

        current_action = self._actions[action]
        agent_x = min(max(agent_x + current_action[0],0),self._screen_height-1)
        agent_y = min(max(agent_y + current_action[1],0),self._screen_width-1)

        reward, done = 0.0, False
        if (agent_x == target_x) and (agent_y == target_y):
            reward = 1.0
            self._score = self._score + reward
            # New target
            target_x, target_y = 0, 0
            while (target_x==0) and (target_y==0):
                target_x = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
                target_y = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
            target_x = target_x + agent_x
            target_y = target_y + agent_y
            if (target_x <0 or target_x >= self._screen_height) or (target_y < 0 or target_y >= self._screen_width):
                agent_x, agent_y = 5, 5
                target_x, target_y = 0, 0
                while (target_x==0) and (target_y==0):
                    target_x = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
                    target_y = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
                target_x = target_x + agent_x
                target_y = target_y + agent_y

        self._state = [agent_x, agent_y, target_x, target_y, current_action]
        if (self._time >= self._max_timesteps):
            done = True

        # Sending the external stimulation over TCP port
        if self._tcp_tagging:
            padding=[0]*8
            event_id = [0, 0, 0, agent_prev_x, agent_prev_y, target_prev_x, target_prev_y, action]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self._s.sendall(bytearray(padding+event_id+timestamp))

        return self._get_observation(), reward, done, {"ale.lives": self._ale.lives(), "internal_state": self._state}

    def reset(self):
        self._score = 0.0
        self._time = 0
        agent_x, agent_y = 5, 5
        target_x, target_y = 0, 0
        while (target_x==0) and (target_y==0):
            target_x = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
            target_y = (np.random.randint(self._max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
        target_x = target_x + agent_x
        target_y = target_y + agent_y

        self._state = [agent_x, agent_y, target_x, target_y, -1]
        return self._get_observation()


    def _get_observation(self, dstate=0):
        img = np.zeros(self._atari_dims, dtype=np.uint8) # Black screen
        block_width = int(self._atari_width/self._screen_width)
        [agent_x, agent_y, target_x, target_y, action] = self._state

        #Draw blocks
        for idx_x in range(self._screen_height):
            for idx_y in range(self._screen_width):
                img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = 255*(1-arr_cell)
                img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = 255*(1-arr_cell)
                img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = 255*(1-arr_cell)

        #Draw Target
        col_mid = target_y*block_width + block_width/2
        row_mid = target_x*block_width + block_width/2
        img[(self._offset + row_mid-block_width/4):(self._offset + row_mid+block_width/4),col_mid-block_width/4:col_mid+block_width/4,0]=255
        img[(self._offset + row_mid-block_width/4):(self._offset + row_mid+block_width/4),col_mid-block_width/4:col_mid+block_width/4,1]=0
        img[(self._offset + row_mid-block_width/4):(self._offset + row_mid+block_width/4),col_mid-block_width/4:col_mid+block_width/4,2]=0
        # idx_x, idx_y = target_x, target_y
        # img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 2] = (1-arr_target)*255
        # img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 1] = (1-arr_target)*255
        # img[(self._offset + idx_x*block_width):(self._offset + (idx_x+1)*block_width), idx_y*block_width:(idx_y+1)*block_width, 0] = (1-arr_target)*255

        # Draw Cursor
        x_c = agent_x*block_width + self._offset
        y_c = agent_y*block_width

        for idx_x in range(arr_ghost.shape[0]):
            for idx_y in range(arr_ghost.shape[1]):
                if arr_ghost[idx_x, idx_y] == 0:    # Boundary
                    img[x_c+idx_x, y_c+idx_y, :] = 0
                elif arr_ghost[idx_x, idx_y] == 1:    # Body
                    img[x_c+idx_x, y_c+idx_y, 2] = 255
                    img[x_c+idx_x, y_c+idx_y, 0] = 0
                    img[x_c+idx_x, y_c+idx_y, 1] = 0
                    # img[x_c+idx_x, y_c+idx_y, 1] = 0
                elif arr_ghost[idx_x, idx_y] == 2:    # Eyes
                    img[x_c+idx_x, y_c+idx_y, 1] = 0
                    img[x_c+idx_x, y_c+idx_y, 0] = 0
                    img[x_c+idx_x, y_c+idx_y, 2] = 0
                else:   # background
                    img[x_c+idx_x, y_c+idx_y, :] = 255


        return img

    def render(self, mode='human', close=False, dstate=0):
        img = self._get_observation(dstate)
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=1920)
            self.viewer.imshow(np.repeat(np.repeat(img, 5, axis=0), 5, axis=1))
            return self.viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(GridEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self._tcp_tagging:
            self.s.close()

    def _get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self.action_set]

    @property
    def _n_actions(self):
        return len(self.action_set)

    def seed(self, seed=None):
        self._np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    # A function for TCP_tagging in openvibe
    # transform a value into an array of byte values in little-endian order.
    def to_byte(self, value, length):
        for x in range(length):
            yield value%256
            value//=256


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self._get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


ACTION_MEANING = {
    0 : "UP",
    1 : "RIGHT",
    2 : "LEFT",
    3 : "DOWN",
}

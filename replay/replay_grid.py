#### Written and Copyright by Mohit Agarwal
#### Georgia Institute of Technology
#### Email: me.agmohit@gmail.com

### Code to replay the game for recorded error-potentials


### 16 Electrode Channels: ['Pz', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'Cz', 'Fz', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'Fpz']
### Agent took random action with Perr over 4 actions namely: Up, Right, Left and Down
### Grid size was 10x10 horizontal and vertical blocks
### Stimulations are as: [0, 0, 0, agent_x, agent_y, ghost_x, ghost_y, action]
### actions are [0,1,2,3] move up, right, left and down : state is the new updated state after taking an action
### Data was recorded for Duo, Yubing, Mohit and Shruti

### If want to classify data for every subject individually - put separate folder for every subject
### If training and testing on different files, the folder structure should be similar in both train and test

import gym
import gym_grid
import time
import numpy as np
import os
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as patches

file_path = "replay/"

stim_nonErrp = 1
stim_Errp = 0
stim_skip = -1


def convert_codes(x):
    if ':' in x:
        return x.split(':',1)[1]
    else:
        return (x.strip() or float('NaN'))

# function to convert stimulations
def to_byte(value, length):
    for x in range(length):
        yield value%256
        value//=256


def replace_errp_stim(stim):
    stimulation = []
    for stim_id, stim_code in enumerate(stim[:,1]):
        if (not math.isnan(stim_code)) and stim_code!=33552 and stim_code!=33553 and stim_code!=33554:
            [a, b, c, agent_prev_x, agent_prev_y, target_prev_x, target_prev_y, action] = list(to_byte(stim_code, 8))
            agent_prev_x, agent_prev_y = int(agent_prev_x), int(agent_prev_y)
            target_prev_x, target_prev_y = int(target_prev_x), int(target_prev_y)
            action_val = int(action)
            assert a+b+c==0, "problem in decoding stimulation"
            # if (a+b+c != 0):
            print stim_code, a, b, c, agent_prev_x, agent_prev_y, target_prev_x, target_prev_y, action
			#Correct or Incorrect
            if action_val == 0: # Up
                if agent_prev_x == 0:
                    new_stim = stim_skip
                elif target_prev_x < agent_prev_x:
                    new_stim = stim_nonErrp
                else:
                    new_stim = stim_Errp
            elif action_val == 1: # Right
                if agent_prev_y == 9:
                    new_stim = stim_skip
                elif target_prev_y > agent_prev_y:
                    new_stim = stim_nonErrp
                else:
                    new_stim = stim_Errp
            elif action_val == 2: # Left
                if agent_prev_y == 0:
                    new_stim = stim_skip
                elif target_prev_y < agent_prev_y:
                    new_stim = stim_nonErrp
                else:
                    new_stim = stim_Errp
            elif action_val == 3: # Down
                if agent_prev_x == 9:
                    new_stim = stim_skip
                elif target_prev_x > agent_prev_x:
                    new_stim = stim_nonErrp
                else:
                    new_stim = stim_Errp
            else:
                print "ssup"
            stimulation.append([agent_prev_x, agent_prev_y, target_prev_x, target_prev_y, action, new_stim])
    return np.array(stimulation)

def loadData(DataFolder, curr_file):
    raw_EEG = np.loadtxt(open(os.path.join(DataFolder,curr_file), "rb"), delimiter=",", skiprows=1, usecols=(0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
    stim = np.loadtxt(open(os.path.join(DataFolder,curr_file), "rb"), delimiter=",", skiprows=1, usecols=(0,21), converters={21: convert_codes})
    stim_time = np.loadtxt(open(os.path.join(DataFolder,curr_file), "rb"), delimiter=",", skiprows=1, usecols=(0,22), converters={22: convert_codes})
    stim = replace_errp_stim(stim)
    return raw_EEG, stim, stim_time

env = gym.make('grid-v0')
env.reset()

file_name = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))][0]
print "file name: " , file_name

EEG, stim, stim_time = loadData(file_path, file_name)

fig  = plt.figure(figsize = (8,8))
gridspec.GridSpec(9,8)

ax1 = plt.subplot2grid((9,8), (0,0), colspan=8, rowspan=8)
ax2 = plt.subplot2grid((9,8), (8,0), colspan=8, rowspan=1)



img = env.unwrapped._get_observation()
img = np.repeat(np.repeat(img, 5, axis=0), 5, axis=1)
im = ax1.imshow(img, animated=True)

p = patches.Rectangle((0.0, 0.0), .1, 1, color="grey")
ax2.add_patch(p)
ax2.text(0.5*(1), 0.5*(1), env.unwrapped._state, horizontalalignment='center', verticalalignment='center', fontsize=20, color='grey')

idx = -1
def updatefig(*args):
	global idx
	idx = idx + 1
	if idx < len(stim):
		# Game Plot
		curr_state =  [int(stim[idx,0]), int(stim[idx,1]), int(stim[idx,2]), int(stim[idx,3]), int(stim[idx,4])]
		env.unwrapped._state = curr_state
		img = env.unwrapped._get_observation()
		img = np.repeat(np.repeat(img, 5, axis=0), 5, axis=1)
		im.set_array(img)

		if int(stim[idx,5]) == stim_Errp:
			color = "red"
		elif int(stim[idx,5]) == stim_nonErrp:
			color = "green"
		else:
			color = "black"
		p = patches.Rectangle((0.0, 0.0), .1, 1, color=color)
		ax2.clear()
		ax2.text(0.5*(1), 0.5*(1), env.unwrapped._state, horizontalalignment='center', verticalalignment='center', fontsize=20, color=color)
		ax2.add_patch(p)
		ax2.figure.canvas.draw()


		return [im]

ani = animation.FuncAnimation(fig, updatefig, interval=1000)
env.close()
plt.show()

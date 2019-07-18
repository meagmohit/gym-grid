import gym
import gym_grid
import time
import numpy as np

env = gym.make('GridNoFrameskip-v3')

env.reset()
action_set = [0,1,2,3]
p_err = 0.2
speed = 1.0 # in seconds must be greater than 0.2

# Action Codes: 0,1,2,3 : Up, Right, Left and Down respectively

for _ in range(1):
    done = False
    env.reset()
    env.render()
    time.sleep(10)
    while not done:
        state = env.unwrapped._state
        [agent_x, agent_y, target_x, target_y, action] = state
        correct_action, incorrect_action = [], []
        if target_x > agent_x:
            # DOWN
            correct_action.append(3)
            incorrect_action.append(0)
        elif target_x < agent_x:
            # UP
            correct_action.append(0)
            incorrect_action.append(3)
        else:
            incorrect_action.append(0)
            incorrect_action.append(3)
        if target_y > agent_y:
            # RIGHT
            correct_action.append(1)
            incorrect_action.append(2)
        elif target_y < agent_y:
            # LEFT
            correct_action.append(2)
            incorrect_action.append(1)
        else:
            incorrect_action.append(2)
            incorrect_action.append(1)
        print state
        action = np.random.choice(['C', 'IC'], p = [1-p_err, p_err])
        if action == 'C':
            action = np.random.choice(correct_action)
        else:
            action = np.random.choice(incorrect_action)
        observation, reward, done, info = env.step(action)
        print action, reward
        env.render()
        time.sleep(speed)
        time.sleep(np.random.random()/4)   # Generates random number between 0 and 0.25

env.close()

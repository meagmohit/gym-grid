# gym-maze
A simple 2-D grid game (with atari rendering) in the gym OpenAI environment

Important: Need to properly check before using this repository : Not maintaned
<p align="center">
  <img src="extras/grid_screenshot.png" width="150" title="Screenshot of Maze Game">
</p>

## Installation instructions
----------------------------

Requirements: gym with atari dependency

```shell
git clone https://github.com/meagmohit/gym-grid
cd gym-grid
python setup.py install
```

```python
import gym
import gym_grid
env = gym.make('grid-v0')
env.render()
```

## Environment Details
----------------------

* **grid-v0 :** Default settings (`grid_size=(10,10)`)
* **GridNoFrameskip-v3 :** Default settings (`grid_size=(10,10)`) and `tcp_tagging=True`

## Agent Details
----------------

* `agents/random_agent.py` random agent plays game with given error probability to take actions (Perr).
* `agents/play.py` for humans to play the game. Keys w,s,a,d for UP, Down, Left and Right respectively.
* `agents/agent_recordEEG_openLoop.py`  plays the game slow with TCP connection port to allow data stimulations

## References
-------------

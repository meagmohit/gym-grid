from gym.envs.registration import register

register(
    id='grid-v0',
    entry_point='gym_grid.atari:GridEnv',
)

register(
    id='GridNoFrameskip-v3',
    entry_point='gym_grid.atari:GridEnv',
    kwargs={'tcp_tagging': True}, # A frameskip of 1 means we get every frame
    max_episode_steps=10000,
    nondeterministic=False,
)

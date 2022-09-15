from gym.envs.registration import register
from .envs.simple_arena import SimpleArenaEnv

register(
    id="policy_instances/SimpleArena-v0",
    entry_point=SimpleArenaEnv,
    max_episode_steps=300,
)

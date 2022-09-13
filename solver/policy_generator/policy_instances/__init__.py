from gym.envs.registration import register

register(
    id="policy_instances/SimpleArena-v0",
    entry_point="solver.policy_generator.policy_instances.envs:SimpleArenaEnv",
    max_episode_steps=300,
)

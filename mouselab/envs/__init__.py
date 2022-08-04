from mouselab.envs.registry import register
from mouselab.envs.reward_settings import (
    high_decreasing_reward,
    high_increasing_reward,
    large_increasing_reward,
    low_constant_reward,
    normal_env_reward_dict,
)

# Standard environments found in the MCRL / Computational Microscope code

register(
    name="high_increasing",
    branching=[3, 1, 2],
    reward_inputs="depth",
    reward_dictionary=high_increasing_reward,
)

register(
    name="high_decreasing",
    branching=[3, 1, 2],
    reward_inputs="depth",
    reward_dictionary=high_decreasing_reward,
)

register(
    name="low_constant",
    branching=[3, 1, 2],
    reward_inputs="depth",
    reward_dictionary=low_constant_reward(3),
)

register(
    name="large_increasing",
    branching=[3, 1, 1, 2, 3],
    reward_inputs="depth",
    reward_dictionary=large_increasing_reward,
)

# Environments found in an old mouselab/mouselab_utils.py version

register(
    name="constant_high",
    branching=[4, 1, 2],
    reward_inputs="depth",
    reward_dictionary=normal_env_reward_dict("constant_high"),
)

register(
    name="increasing",
    branching=[4, 1, 2],
    reward_inputs="depth",
    reward_dictionary=normal_env_reward_dict("increasing"),
)

register(
    name="decreasing",
    branching=[4, 1, 2],
    reward_inputs="depth",
    reward_dictionary=normal_env_reward_dict("decreasing"),
)

register(
    name="constant_low",
    branching=[4, 1, 2],
    reward_inputs="depth",
    reward_dictionary=normal_env_reward_dict("constant_low"),
)

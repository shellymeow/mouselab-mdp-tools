from mouselab.distributions import Categorical, Normal

high_increasing_reward = {
    1: Categorical([-4, -2, 2, 4]),
    2: Categorical([-8, -4, 4, 8]),
    3: Categorical([-48, -24, 24, 48]),
}

high_decreasing_reward = {
    1: Categorical([-48, -24, 24, 48]),
    2: Categorical([-8, -4, 4, 8]),
    3: Categorical([-4, -2, 2, 4]),
}

toy_imbalanced_reward = {
    1: Categorical([-4, -2, 2, 4]),
    3: Categorical([-8, -4, 4, 8]),
    2: Categorical([-8, -4, 4, 8]),
    4: Categorical([-48, -24, 24, 48]),
}

def low_constant_reward(depth):
    return {depth: Categorical([-10, -5, 5, 10]) for depth in range(1, depth + 1)}


large_increasing_reward = {
    level_idx + 1: Normal(0, level) for level_idx, level in enumerate([1, 2, 4, 8, 32])
}


def normal_env_reward_dict(variance_structure):
    sigmas = {
        "constant_high": [0, 20, 20, 20],
        "increasing": [0, 2, 4, 20],
        "decreasing": [0, 20, 10, 5],
        "constant_low": [0, 1, 1, 1],
    }

    def reward(depth):
        if depth > 0:
            return Normal(0, sigmas[variance_structure][depth]).to_discrete(6)
        return 0.0

    return {depth: reward(depth) for depth in range(0, 4)}
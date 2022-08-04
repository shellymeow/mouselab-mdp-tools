import numpy as np
import pytest
from contexttimer import Timer
from exact_implementations.exact_mem import solve_mem

from mouselab.env_utils import get_all_possible_ground_truths
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_increasing_reward
from mouselab.exact import solve
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

"""
This file exists to experiment with possible ways to run exact.py and save Q functions.
It primarily runs a few solution functions and outputs the mean and standard deviation time.
"""  # noqa: E501

exact_timing_test_data = [
    {
        "env": {
            "name": "small_solve",
            "branching": [2, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        },
        "kwargs": [
            {},
            {},
            {"verbose": False, "save_q": True},
            {"verbose": False, "save_q": True},
        ],
    },
    {
        "env": {
            "name": "medium_solve",
            "branching": [2, 1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        },
        "kwargs": [
            {},
            {},
            {"verbose": False, "save_q": True},
            {"verbose": False, "save_q": True},
        ],
    },
]


@pytest.fixture(params=exact_timing_test_data)
def exact_timing_test_cases(request):
    register(**request.param["env"])

    def get_selected_ground_truths(env_setting):
        env = MouselabEnv.new_symmetric_registered(env_setting)

        # get approximately 100 ground truths
        ground_truths = list(get_all_possible_ground_truths(env))
        skip = len(ground_truths) // 100
        selected_ground_truths = ground_truths[::skip]
        return selected_ground_truths

    # get 100 ground truths for test env
    selected_ground_truths = get_selected_ground_truths(request.param["env"]["name"])

    request.param["kwargs"][2]["ground_truths"] = selected_ground_truths

    yield request.param["env"]["name"], request.param["kwargs"]


def time_solve(env_setting, solve_function, **solve_kwargs):
    """
    Runs solve function, with solve kwargs, for env_setting
    """
    env = MouselabEnv.new_symmetric_registered(env_setting)
    with Timer() as t:
        Q, V, pi, info = solve_function(env, **solve_kwargs)
        V(env.init)
        elapsed = t.elapsed
    return elapsed


@pytest.mark.skip(reason="Very slow, just here for timing reasons.")
def test_exact_timing(exact_timing_test_cases):
    env_setting, solve_kwargs_list = exact_timing_test_cases
    solve_functions = [solve, solve_mem, timed_solve_env, timed_solve_env]
    num_repetitions = 10

    for solve_function, solve_kwargs in zip(solve_functions, solve_kwargs_list):
        print(solve_function.__name__, solve_kwargs.keys())
        ts = [
            time_solve(env_setting, solve_function, **solve_kwargs)
            for _ in range(num_repetitions)
        ]
        print(np.mean(ts), np.std(ts))
        assert True

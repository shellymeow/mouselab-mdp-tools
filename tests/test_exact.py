import pytest

from mouselab.distributions import Categorical
from mouselab.env_utils import get_all_possible_sa_pairs_for_env
from mouselab.envs.registry import register
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv

# set up test cases
exact_test_case_data = [
    {
        "env": {
            "name": "small_test_case",
            "branching": [1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": {1: Categorical([-500]), 2: Categorical([-60, 60])},
        }
    },
    {
        "env": {
            "name": "medium_test_case",
            "branching": [1, 2, 1],
            "reward_inputs": "depth",
            "reward_dictionary": {
                1: Categorical([-500]),
                2: Categorical([-60, 60]),
                3: Categorical([-90, 90]),
            },
        }
    },
]


@pytest.fixture(params=exact_test_case_data)
def test_env(request):
    register(**request.param["env"])

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"], cost=1)


def test_sequence(test_env):
    Q, V, pi, info = timed_solve_env(test_env, verbose=False, save_q=True)
    sa_pairs = get_all_possible_sa_pairs_for_env(test_env)

    # tests if dictionary keys are same as all possible (state,action) pairs
    assert set(sa_pairs) == set(info["q_dictionary"].keys())

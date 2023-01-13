import pytest

from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv
from mouselab.graph_utils import get_structure_properties
import numpy as np

from mouselab.policies import DiscountedSoftmaxPolicy, DiscountedOptimalQ
from mouselab.agents import Agent

# set up test case
exact_test_case_data = [{
        "env": {
            "name": "small_test_case",
            "branching": [1, 1],
            "reward_inputs": "depth",
            "reward_dictionary": {1: Categorical([-500]), 2: Categorical([-60, 60])},
        },
        "structure": {
            "layout" : {"0": [0, 0], "1": [0, -1], "2": [1, -1]},
            "initial": "0",
            "graph"  : {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"]},
                "2": {},
            },
        },
    }]

@pytest.fixture(params=exact_test_case_data)
def test_env(request):
    register(**request.param["env"])

    env_params = {"cost":1,
                      "include_last_action":False,
                      "mdp_graph_properties":get_structure_properties(request.param["structure"])}

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"],
                                               **env_params)

def test_sequence(test_env):
    Q, V, pi, info = timed_solve_env(test_env, verbose=False, save_q=True)

    # TODO: refactor these into test cases
    ## Test action probabilities
    # 1. Low temperature softmax, no discount
    agent = Agent()
    agent.register(test_env)

    pol = DiscountedSoftmaxPolicy(preference=Q, gamma=1)
    agent.register(pol)

    assert(np.allclose(agent.policy.action_distribution(test_env._state), [0,0,0,1]))

    # 2. High temperature softmax, no discount
    agent = Agent()
    agent.register(test_env)

    pol = DiscountedSoftmaxPolicy(preference=Q, gamma=1, temp=5e10)
    agent.register(pol)

    assert(np.allclose(agent.policy.action_distribution(test_env._state), [0, 1/3, 1/3, 1/3]))

    # 3. Optimal, no discount
    agent = Agent()
    agent.register(test_env)

    pol = DiscountedOptimalQ(preference=Q, gamma=1)
    agent.register(pol)

    assert(np.allclose(agent.policy.action_distribution(test_env._state),[0,0,0,1]))

    ## Test preferences
    # 1. Softmax, no discount, preferences
    pol = DiscountedSoftmaxPolicy(preference=Q, gamma=1)
    agent.register(pol)

    assert (np.allclose(agent.policy.preferences(test_env._state)[0], [-501, -501, -500]))

    # 2. Softmax, .90 discount starting at depth = 1, preferences
    pol = DiscountedSoftmaxPolicy(preference=Q, gamma=1, discount_first_step=True)
    agent.register(pol)

    assert (np.allclose(agent.policy.preferences(test_env._state)[0], [-501*.9, -501*.9*.9, -500]))

    # 2. Softmax, .90 discount starting at depth = 2, preferences
    pol = DiscountedSoftmaxPolicy(preference=Q, gamma=1)
    agent.register(pol)

    assert (np.allclose(agent.policy.preferences(test_env._state)[0], [-501, -501*.9, -500]))

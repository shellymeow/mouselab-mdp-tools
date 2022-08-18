import pytest

from mouselab.cost_functions import distance_graph_cost
from mouselab.distributions import Categorical
from mouselab.env_utils import get_all_possible_sa_pairs_for_env
from mouselab.envs.registry import register
from mouselab.exact_utils import timed_solve_env
from mouselab.mouselab import MouselabEnv
from mouselab.graph_utils import get_structure_properties
from mouselab.exact import hash_tree
# set up test cases
exact_test_case_data = [
    {
        "env": {
            "name": "small_test_case",
            "branching": [1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": {1: Categorical([-500]), 2: Categorical([-60, 60])},
        },
        "structure": {
            "layout" : {"0": [0, 0], "1": [0, -1], "2": [1, -1], "3": [-1, -1]},
            "initial": "0",
            "graph"  : {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"], "left": [0, "3"]},
                "2": {},
                "3": {},
            },
        },
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
            }
        },
        "structure": {
            "layout" : {
                "0": [0, 0],
                "1": [0, -1],
                "2": [1, -1],
                "3": [2, -1],
                "4": [-1, -1],
                "5": [-2, -1],
            },
            "initial": "0",
            "graph"  : {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"], "left": [0, "4"]},
                "2": {"right": [0, "3"]},
                "3": {"left": [0, "5"]},
                "4": {},
                "5": {},
            },
        },
    },
]


@pytest.fixture(params=exact_test_case_data)
def test_env(request):
    register(**request.param["env"])

    env_params = {"cost":distance_graph_cost(max_penalty=None, distance_multiplier=1, given_cost=0),
                      "include_last_action":True,
                      "mdp_graph_properties":get_structure_properties(request.param["structure"])}

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"],
                                               **env_params)


def test_sequence(test_env):
    Q, V, pi, info = timed_solve_env(test_env, verbose=False, save_q=True)
    sa_pairs = get_all_possible_sa_pairs_for_env(test_env)

    # tests if dictionary keys are same as all possible (state,action) pairs
    assert set(sa_pairs) == set(info["q_dictionary"].keys())

@pytest.fixture(params=[{**exact_test_case_data[0], "manual": manual_data} for manual_data in [[((0, -500, Categorical([60,-60]), -60, 1), 2), -501],[((0, Categorical([-500]), Categorical([60,-60]), Categorical([60,-60]), 0), 2), -470 - (2)**(1/2)]]])
def fixture_manual_distance(request):
    register(**request.param["env"])

    if "structure" in request.param:
        env_params = {"cost":distance_graph_cost(max_penalty=None, distance_multiplier=1, given_cost=0),
                      "include_last_action":True,
                      "mdp_graph_properties":get_structure_properties(request.param["structure"])}
    else:
        env_params = {}

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"],
                                               **env_params), request.param["manual"]

def test_manual_distance(fixture_manual_distance):
    test_env, test_case = fixture_manual_distance
    sa, q_value = test_case

    Q, V, pi, info = timed_solve_env(test_env, verbose=False, save_q=True)
    assert Q(*sa)==q_value


def test_exact_extended(test_env):
    _, _, _ , info = timed_solve_env(test_env, verbose=False, save_q=True, hash_state="test")
    _, _, _, info_mem = timed_solve_env(test_env, verbose=False, save_q=True)

    assert info["q_dictionary"]==info_mem["q_dictionary"]



extended_belief_state_test_case = [
    {
        "env"      : {
            "name"             : "small_test_case",
            "branching"        : [1, 2],
            "reward_inputs"    : "depth",
            "reward_dictionary": {1: Categorical([-500]), 2: Categorical([-60, 60])},
        },
        "structure": {
            "layout" : {"0": [0, 0], "1": [0, -1], "2": [1, -1], "3": [-1, -1]},
            "initial": "0",
            "graph"  : {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"], "left": [0, "3"]},
                "2": {},
                "3": {},
            },
        },
        "click_orders" : click_order
    } for click_order in [[(1,2),(2,1)],[(3,1),(1,3)]]
] + [
    {
        "env"      : {
            "name"             : "medium_test_case",
            "branching"        : [1, 2, 1],
            "reward_inputs"    : "depth",
            "reward_dictionary": {
                1: Categorical([-500]),
                2: Categorical([-60, 60]),
                3: Categorical([-90, 90]),
            }
        },
        "structure": {
            "layout" : {
                "0": [0, 0],
                "1": [0, -1],
                "2": [1, -1],
                "3": [2, -1],
                "4": [-1, -1],
                "5": [-2, -1],
            },
            "initial": "0",
            "graph"  : {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"], "left": [0, "4"]},
                "2": {"right": [0, "3"]},
                "3": {"left": [0, "5"]},
                "4": {},
                "5": {},
            },
        },
        "click_orders": click_order
    } for click_order in [[(2,4,5), (2,5,4)]]
]


@pytest.fixture(params=extended_belief_state_test_case)
def setup_extended_belief_state(request):
    """
    Would be better to refactor this since test_exact.py has similar function
    """
    register(**request.param["env"])

    env_params = {"cost":distance_graph_cost(max_penalty=None, distance_multiplier=1),
                      "include_last_action":True,
                      "mdp_graph_properties":get_structure_properties(request.param["structure"])}

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"],
                                               **env_params), request.param["click_orders"]


def test_extended_belief_state_hash(setup_extended_belief_state):
    """
    Test to see if hash includes last action for extended belief state
    """
    env, click_orders = setup_extended_belief_state
    hash_state = lambda state: hash_tree(env, state)

    final_hashes = []
    for click_order in click_orders:
        env.reset()
        for action in click_order:
            env.step(action)
        final_hashes.append(hash_state(env._state))

    # all should be unique
    assert len(set(final_hashes))==len(final_hashes)


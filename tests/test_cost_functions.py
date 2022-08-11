import itertools

import pytest

from mouselab.cost_functions import distance_graph_cost, linear_depth, side_cost
from mouselab.distributions import Categorical
from mouselab.envs.registry import register, registry
from mouselab.graph_utils import get_structure_properties
from mouselab.mouselab import MouselabEnv

env_params = [
    {
        "env": {
            "name": "small_test_case",
            "branching": [1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": {1: Categorical([-500]), 2: Categorical([-60, 60])},
        },
        "structure": {
            "layout": {"0": [0, 0], "1": [0, -1], "2": [1, -1], "3": [-1, -1]},
            "initial": "0",
            "graph": {
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
            },
        },
        "structure": {
            "layout": {
                "0": [0, 0],
                "1": [0, -1],
                "2": [1, -1],
                "3": [-1, -1],
                "4": [2, -1],
                "5": [-2, -1],
            },
            "initial": "0",
            "graph": {
                "0": {"up": [0, "1"]},
                "1": {"right": [0, "2"], "left": [0, "3"]},
                "2": {"right": [0, "4"]},
                "3": {"left": [0, "5"]},
                "4": {},
                "5": {},
            },
        },
    },
    {
        "env": {"name": "high_increasing"},
        "structure": {
            "layout": {
                "0": [0, 0],
                "1": [0, -1],
                "2": [0, -2],
                "3": [1, -2],
                "4": [-1, -2],
                "5": [1, 0],
                "6": [2, 0],
                "7": [2, -1],
                "8": [2, 1],
                "9": [-1, 0],
                "10": [-2, 0],
                "11": [-2, -1],
                "12": [-2, 1],
            },
            "initial": "0",
            "graph": {
                "0": {"up": [0, "1"], "right": [0, "5"], "left": [0, "9"]},
                "1": {"up": [0, "2"]},
                "2": {"right": [0, "3"], "left": [0, "4"]},
                "3": {},
                "4": {},
                "5": {"right": [0, "6"]},
                "6": {"up": [0, "7"], "down": [0, "8"]},
                "7": {},
                "8": {},
                "9": {"left": [0, "10"]},
                "10": {"up": [0, "11"], "down": [0, "12"]},
                "11": {},
                "12": {},
            },
        },
    },
]

depth_params = [
    [linear_depth(0, 1), [-1, -2, -3]],
    [linear_depth(1, 1), [-2, -3, -4]],
    [linear_depth(1, 0), [-1, -1, -1]],
]


@pytest.fixture(
    params=[
        [product[0], *product[1]]
        for product in itertools.product(env_params, depth_params)
    ]
)
def depth_cost_test_cases(request):
    if request.param[0]["env"]["name"] not in registry.envs:
        register(**request.param[0]["env"])

    # hand coded depths for each experiment setting
    if request.param[0]["env"]["name"] == "high_increasing":
        depth_result = {
            1: request.param[2][0],
            5: request.param[2][0],
            9: request.param[2][0],
            2: request.param[2][1],
            6: request.param[2][1],
            10: request.param[2][1],
            3: request.param[2][2],
            4: request.param[2][2],
            7: request.param[2][2],
            8: request.param[2][2],
            11: request.param[2][2],
            12: request.param[2][2],
        }
    elif request.param[0]["env"]["name"] == "small_test_case":
        depth_result = {
            1: request.param[2][0],
            2: request.param[2][1],
            3: request.param[2][1],
        }
    elif request.param[0]["env"]["name"] == "medium_test_case":
        depth_result = {
            1: request.param[2][0],
            2: request.param[2][1],
            3: request.param[2][2],
            4: request.param[2][1],
            5: request.param[2][2],
        }
    else:
        raise NotImplementedError

    yield request.param[0]["env"]["name"], get_structure_properties(
        request.param[0]["structure"]
    ), request.param[1], depth_result


def test_depth_cost(depth_cost_test_cases):
    (
        experiment_setting,
        mdp_graph_properties,
        cost_function,
        depth_result,
    ) = depth_cost_test_cases
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting,
        mdp_graph_properties=mdp_graph_properties,
        cost=cost_function,
    )

    cost_function_result = {}
    for node in depth_result.keys():
        cost_function_result[node] = env.cost(node)

    assert depth_result == cost_function_result


distances = {
    "small_test_case": {(0, 1): 1, (2, 3): 2, (3, 2): 2, (0, 3): 2 ** (1 / 2)},
    "medium_test_case": {
        (4, 5): 4,
        (5, 4): 4,
        (2, 4): 1,
        (4, 2): 1,
        (0, 3): 2 ** (1 / 2),
    },
    "high_increasing": {
        (0, 1): 1,
        (3, 0): 5 ** (1 / 2),
        (0, 2): 2,
        (0, 3): 5 ** (1 / 2),
        (2, 3): 1,
        (3, 2): 1,
        (8, 3): 10 ** (1 / 2),
        (3, 8): 10 ** (1 / 2),
        (4, 8): 18 ** (1 / 2),
        (8, 4): 18 ** (1 / 2),
    },
}

distance_parameters = []
for env_param in env_params:
    for states, result in distances[env_param["env"]["name"]].items():
        distance_parameters.append([env_param, states, result + 1])


@pytest.fixture(params=distance_parameters)
def distance_cost_test_cases(request):
    if request.param[0]["env"]["name"] not in registry.envs:
        register(**request.param[0]["env"])

    yield request.param[0]["env"]["name"], get_structure_properties(
        request.param[0]["structure"]
    ), request.param[1], request.param[2]


def test_distance_cost(distance_cost_test_cases):
    (
        experiment_setting,
        mdp_graph_properties,
        states,
        cost_output,
    ) = distance_cost_test_cases

    cost_function = distance_graph_cost(max_penalty=None, distance_multiplier=1)
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting,
        mdp_graph_properties=mdp_graph_properties,
        cost=cost_function,
        include_last_action=True,
    )

    if states[0] > 0:
        env.step(states[0])
    assert env.cost(states[1]) == -cost_output
    # can only step if next action is greater than 0
    # we have a test case where it is 0 to check symmetry
    if states[1] > 0:
        _, reward, _, _ = env.step(states[1])
        assert reward == -cost_output


@pytest.mark.parametrize(
    "experiment_setting,mdp_graph_properties,side_dict",
    [
        [
            env_params[2]["env"]["name"],
            get_structure_properties(env_params[2]["structure"]),
            {
                1: -1,
                2: -1,
                3: -1,
                4: -1,
                5: -1.5,
                6: -1.5,
                7: -1.5,
                8: -1.5,
                9: -0.5,
                10: -0.5,
                11: -0.5,
                12: -0.5,
            },
        ]
    ],
)
def test_side_cost(experiment_setting, mdp_graph_properties, side_dict):
    env = MouselabEnv.new_symmetric_registered(
        experiment_setting,
        mdp_graph_properties=mdp_graph_properties,
        cost=side_cost(1, {"right": 1 / 2, "up": 1 / 3, "left": 1 / 6}),
    )

    constructed_side_costs = {}
    for node in side_dict.keys():
        constructed_side_costs[node] = env.cost(node)

    assert constructed_side_costs == side_dict

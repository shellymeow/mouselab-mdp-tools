import hashlib

import numpy as np
import pytest

from mouselab.distributions import Categorical
from mouselab.envs.registry import register
from mouselab.envs.reward_settings import high_increasing_reward
from mouselab.exact import hash_tree
from mouselab.mouselab import MouselabEnv

test_env_data = [
    {
        "env": {
            "name": "old_cat_hi",
            "branching": [3, 1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        }
    },
    {
        "env": {
            "name": "new_cat_hi",
            "branching": [3, 1, 2],
            "reward_inputs": "depth",
            "reward_dictionary": high_increasing_reward,
        },
    },
]


@pytest.fixture(params=test_env_data)
def test_env(request):
    # Two different hashing functions -- one as before, one as now
    class OldCategorical(Categorical):
        """
        Categorical with old hashing, equality
        """

        def __init__(self, *args, **kwargs):
            super(OldCategorical, self).__init__(*args, **kwargs)

        def __hash__(self):
            return self._hash

        def __eq__(self, other):
            return hasattr(other, "sample")

    class NewCategorical(Categorical):
        """
        Categorical with hash which can be used in dictionary
        """

        def __init__(self, *args, **kwargs):
            super(NewCategorical, self).__init__(*args, **kwargs)

        def __hash__(self):
            # want the hash to be the same across runs for dictionary keys
            # -> hash str through hashlib
            hash_digest = hashlib.md5(self.__str__().encode("utf-8")).digest()
            return int.from_bytes(hash_digest, "big")

    register(**request.param["env"])

    yield MouselabEnv.new_symmetric_registered(request.param["env"]["name"])


def test_number_hashed_nodes(test_env):
    """
    test Categorical hash is same for same Categorical distribution
    in 3-1-2 should have 4 unique hashed nodes
    """
    assert len(np.unique([hash(state) for state in test_env._state])) == 4


def test_hashing_same(test_env):
    """
    test hash used in exact.py is same for two symmetric states
    """

    # need to convert tuples to states to manipulate
    state_1 = list(test_env._state)
    state_2 = list(test_env._state)

    # -1 and -2 states are both last layer
    state_1[-1] = 48
    state_2[-2] = 48

    assert hash_tree(test_env, tuple(state_2)) == hash_tree(test_env, tuple(state_1))

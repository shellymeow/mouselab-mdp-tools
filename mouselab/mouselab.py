import gym
import numpy as np
from gym import spaces
from pydantic import NonNegativeFloat
from toolz import get, memoize

from numpy.random import default_rng

from mouselab.distributions import PointMass, cmax, expectation, sample, smax
from mouselab.envs.registry import registry
from mouselab.graph_utils import (
    annotate_mdp_graph,
    graph_from_adjacency_list,
)
import networkx as nx
from scipy.spatial import distance

NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache

CACHE_SIZE = int(2 ** 20)
SMALL_CACHE_SIZE = int(2 ** 14)
ZERO = PointMass(0)


class MouselabEnv(gym.Env):
    """MetaMDP for a tree with a discrete unobserved reward function."""

    metadata = {"render.modes": ["human", "array"]}
    term_state = "__term_state__"

    def __init__(
        self,
        tree,
        init,
        ground_truth=None,
        cost=0,
        term_belief=True,
        sample_term_reward=False,
        include_last_action : bool = False,
        power_utility : int = 1,
        seed=None,
        mdp_graph_properties={},
        last_action_info = None
    ):
        """
        :param tree: adjacency list
        :param init: list of distribution behind each node
        :param ground_truth: if known, a list of values behind the nodes
        :param cost: cost for clicking on a node, either a scalar
                    or function implemented similar to those in mouselab.cost_functions
        :param term_belief: TODO
        :param sample_term_belief: TODO
        :param include_last_action: some cost functions depend on the last action,
                    this should generally be initialized as the starting node
        :param seed: seed for numpy random number generator
        :param mdp_graph_properties: properties to add to mdp graph,
                    as dictionary of {attribute : dictionary of {node : value}}
        :param what last_action_info to calculate (only distance supported)
        """
        self.tree = tree
        mdp_graph = graph_from_adjacency_list(self.tree)
        # add properties to mdp_graph for cost function and possible calculating returns
        self.mdp_graph = annotate_mdp_graph(mdp_graph, mdp_graph_properties)

        self.init = (0, *init[1:])

        if ground_truth is not None:
            if len(ground_truth) != len(init):
                raise ValueError("len(ground_truth) != len(init)")
            self.ground_truth = np.array(ground_truth)
            self.ground_truth[0] = 0.0
        else:
            self.ground_truth = np.array(list(map(sample, init)))
            self.ground_truth[0] = 0.0

        self.include_last_action = include_last_action
        self.power_utility = power_utility

        # potentially used for hashing that depends on last action
        if last_action_info == "distance":
            self.last_action_info = self.get_distance_matrix()
        else:
            self.last_action_info = None

        self.rng = default_rng(seed=seed)

        self.term_belief = term_belief
        self.sample_term_reward = sample_term_reward
        self.term_action = len(self.init)

        if hasattr(cost, "__call__"):
            # reads in all graph attributes and last action
            self.cost = lambda state, action: cost(
                state, action, graph=self.mdp_graph) if action != self.term_action else 0
        else:
            # make the cost function return scalar cost for all inputs if not callable
            self.cost = lambda state, action: -abs(cost)

        # Required for gym.Env API.
        self.action_space = spaces.Discrete(len(self.init) + 1)
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=len(self.init))

        self.initial_states = [self.init]
        self.initial_state_probabilities = [1.0]

        if self.include_last_action:
            self.initial_states = [(*initial_state, 0) for initial_state in self.initial_states]

        self.exact = True  # TODO

        self.subtree = self._get_subtree()
        self.subtree_slices = self._get_subtree_slices()
        self.paths = self.get_paths(0)
        self.reset()

        self._hash = hash((str(self.tree), self.init, str(list(self.ground_truth))))

    def __hash__(self):
        return self._hash

    def get_distance_matrix(self, distance_function=distance.euclidean):
        return [[distance_function(self.mdp_graph.nodes[first_node]["layout"], self.mdp_graph.nodes[second_node]["layout"]) for second_node in  self.mdp_graph.nodes(data=False)] for first_node in self.mdp_graph.nodes(data=False)]

    def reset(self):
        return self._reset()

    def _reset(self):
        if len(self.initial_states) > 1:
            self.init = self.rng.choice(
                self.initial_states, p=self.initial_state_probabilities
            )
        self._state = self.init

        # in Val's experiments participants must click on node 0 to begin
        if self.include_last_action:
            self._state = (*self._state, 0)

        return self._state

    def step(self, action):
        return self._step(action)

    def _step(self, action):
        if self._state is self.term_state:
            assert 0, "state is terminal"
        if action == self.term_action:
            reward = self._term_reward()
            self._state = self.term_state
            done = True
        elif not hasattr(self._state[action], "sample"):  # already observed
            assert 0, self._state[action]
            reward = 0
            done = False
        else:  # observe a new node
            reward = self.cost(self._state, action)
            self._state = self._observe(action)
            done = False
        # update last action
        if self.include_last_action and (self._state is not self.term_state):
            # state is tuple, so need to convert to list to modify last action
            state_list = list(self._state)
            # nodes are first len-1 entries, last action is last entry
            self._state = (*state_list[:-1], action)
        return self._state, reward, done, {}

    def _term_reward(self):
        if self.term_belief:
            return self.expected_term_reward(self._state)

        returns = [self.ground_truth[list(path)].sum() for path in self.optimal_paths()]
        if self.sample_term_reward:
            return self.rng.choice(returns)
        else:
            return np.mean(returns)

    def _observe(self, action):
        if self.ground_truth is None:
            result = self._state[action].sample()
        else:
            result = self.ground_truth[action]
        s = list(self._state)
        s[action] = result
        return tuple(s)

    def actions(self, state):
        """Yields actions that can be taken in the given state.

        Actions include observing the value of each unobserved node and terminating.
        """
        if state is self.term_state:
            return
        for i, v in enumerate(state):
            if hasattr(v, "sample"):
                yield i
        yield self.term_action

    def results(self, state, action):
        """Returns a list of possible results of taking action in state.

        Each outcome is (probability, next_state, reward).
        """
        if action == self.term_action:
            yield (1, self.term_state, self.expected_term_reward(state))
        elif self.include_last_action:
            for r, p in state[action]:
                s1 = list(state)
                s1[action] = r
                s1[-1] = action
                yield (p, tuple(s1), self.cost(state, action))
        else:
            for r, p in state[action]:
                s1 = list(state)
                s1[action] = r
                yield (p, tuple(s1), self.cost(state, action))

    def action_features(self, action, state=None):
        state = state if state is not None else self._state
        assert state is not None

        if action == self.term_action:
            return np.array([0, 0, 0, 0, self.expected_term_reward(state)])

        return np.array(
            [
                self.cost(state, action),
                self.myopic_voc(action, state),
                self.vpi_action(action, state),
                self.vpi(state),
                self.expected_term_reward(state),
            ]
        )

    def term_reward(self, state=None):
        """A distribution over the return gained by acting given a belief state."""
        state = state if state is not None else self._state
        return self.node_value(0, state)

    def optimal_paths(self, state=None, tolerance=0.01):
        state = state if state is not None else self._state

        def rec(path):
            children = self.tree[path[-1]]
            if not children:
                yield path
                return
            quals = [self.node_quality(n1, state).expectation() for n1 in children]
            best_q = max(quals)
            for n1, q in zip(children, quals):
                if np.abs(q - best_q) < tolerance:
                    yield from rec(path + (n1,))

        yield from rec((0,))

    @lru_cache(CACHE_SIZE)
    def expected_term_reward(self, state, action=None):
        expected_term_reward = self.term_reward(state).expectation()
        if expected_term_reward >= 0:
            return expected_term_reward ** self.power_utility
        else:
            return - abs(expected_term_reward) ** self.power_utility

    def node_value(self, node, state=None):
        """A distribution over total rewards after the given node."""
        state = state if state is not None else self._state
        return max(
            (self.node_value(n1, state) + state[n1] for n1 in self.tree[node]),
            default=ZERO,
            key=expectation,
        )

    def node_value_to(self, node, state=None):
        """A distribution over rewards up to and including the given node."""
        state = state if state is not None else self._state
        start_value = ZERO
        return sum((state[n] for n in self.path_to(node)), start_value)

    def node_quality(self, node, state=None):
        """A distribution of total expected rewards if this node is visited."""
        state = state if state is not None else self._state
        return self.node_value_to(node, state) + self.node_value(node, state)

    @lru_cache(CACHE_SIZE)
    def myopic_voc(self, action, state) -> NonNegativeFloat:
        if action == self.term_action:
            # no information from final action
            # explicitly set here due to numerical considerations
            return 0
        else:
            return self.node_value_after_observe(
                (action,), 0, state
            ).expectation() - self.expected_term_reward(state)

    @lru_cache(CACHE_SIZE)
    def vpi_branch(self, action, state) -> NonNegativeFloat:
        obs = self._relevant_subtree(action)
        return self.node_value_after_observe(
            obs, 0, state
        ).expectation() - self.expected_term_reward(state)

    # Rewritten to take multiple paths to a node into account
    @lru_cache(CACHE_SIZE)
    def vpi_action(self, action, state) -> NonNegativeFloat:
        if action == self.term_action:
            return 0

        # Unique nodes in all paths leading to the action
        paths = self.path_to(action)
        flat_paths = [node for path in paths for node in path]
        obs = (*self.subtree[action], *flat_paths)
        obs = tuple(np.unique(obs))
        return (self.node_value_after_observe(obs, 0, state).expectation()
                - self.expected_term_reward(state)
                )


    def vpi(self, state, action=None) -> NonNegativeFloat:
        if action == self.term_action:
            return 0

        obs = self.subtree[0]
        return self.node_value_after_observe(
            obs, 0, state
        ).expectation() - self.expected_term_reward(state)

    def unclicked(self, state):
        return sum(1 for x in state if hasattr(x, "sample"))

    def true_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        return r + max((self.true_Q(n1) for n1 in self.tree[node]), default=0)

    def worst_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        return r + min((self.worst_Q(n1) for n1 in self.tree[node]), default=0)

    def rand_Q(self, node):
        """The object-level Q function."""
        r = self.ground_truth[node]
        lst = [self.rand_Q(n1) for n1 in self.tree[node]]
        if lst:
            return r + self.rng.choice(lst)
        return r

    def mean_Q(self, node):
        r = self.ground_truth[node]
        lst = [self.mean_Q(n1) for n1 in self.tree[node]]
        if lst:
            return r + np.mean(lst)
        return r

    def get_paths(self, node):
        if self.tree[node] == []:
            return [[]]
        paths = []
        for n in self.tree[node]:
            new_paths = self.get_paths(n)
            for path in new_paths:
                path.insert(0, n)
                paths.append(path)
        return paths

    @lru_cache(None)
    def _relevant_subtree(self, node):
        trees = [self.subtree[n1] for n1 in self.tree[0]]
        for t in trees:
            if node in t:
                return tuple(t)
        assert False

    @lru_cache(None)
    def leaves(self):
        return [path[-1] for path in self.all_paths()]

    def path_values(self, state):
        return [self.node_quality(node, state) for node in self.leaves()]

    def node_value_after_observe(self, obs, node, state):
        """A distribution over the expected value of node, after making an observation.

        obs can be a single node, a list of nodes, or 'all'
        """
        obs_tree = self.to_obs_tree(state, node, obs)
        if self.exact:
            return exact_node_value_after_observe(obs_tree)
        else:
            return node_value_after_observe(obs_tree)

    @memoize
    def path_to(self, node, start=0):
        path = [start]
        if node == start:
            return path
        for _ in range(10000):
            children = self.tree[path[-1]]
            for i, child in enumerate(children):
                if child == node:
                    path.append(node)
                    return path
                if child > node:
                    path.append(children[i - 1])
                    break
            else:
                path.append(child)
        assert False

    @memoize
    def all_paths(self, start=0):
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return list(rec([start]))

    def _get_subtree_slices(self):
        slices = [0] * len(self.tree)

        def get_end(n):
            end = max((get_end(n1) for n1 in self.tree[n]), default=n + 1)
            slices[n] = slice(n, end)
            return end

        get_end(0)
        return slices

    def _get_subtree(self):
        def gen(n):
            yield n
            for n1 in self.tree[n]:
                yield from gen(n1)

        return [tuple(gen(n)) for n in range(len(self.tree))]

    @staticmethod
    def branching_and_reward_to_inputs(branching, reward):
        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        tree = []

        def expand(d):
            my_idx = len(init)
            init.append(reward(d))
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d + 1)
                children.append(child_idx)
            return my_idx

        expand(0)
        return tree, init

    @classmethod
    def new_symmetric(cls, branching, reward, seed=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure."""
        tree, init = cls.branching_and_reward_to_inputs(branching=branching, reward=reward)

        return cls(tree, init, seed=seed, **kwargs)

    @classmethod
    def new_symmetric_registered(cls, experiment_setting, seed=None, **kwargs):
        if registry(experiment_setting).reward_inputs != "depth":
            raise ValueError("Symmetric can only be used if reward input is depth.")

        branching = registry(experiment_setting).branching
        reward = registry(experiment_setting).reward_function

        return MouselabEnv.new_symmetric(branching, reward, seed=seed, **kwargs)

    def symmetric_branching(branching):
        init = []
        tree = []

        def expand(d):
            my_idx = len(init)
            init.append(None)
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d + 1)
                children.append(child_idx)
            return my_idx

        expand(0)

        return tree, init

    @classmethod
    def new_registered(cls, experiment_setting, seed=None, **kwargs):
        if registry(experiment_setting).reward_inputs == "depth":
            return MouselabEnv.new_symmetric_registered(experiment_setting, seed=seed, **kwargs)
        elif registry(experiment_setting).reward_inputs == "node":
            branching = registry(experiment_setting).branching
            tree, init = MouselabEnv.symmetric_branching(branching)

            reward = registry(experiment_setting).reward_function
            init = [reward(node) for node in range(len(init))]
            return cls(tree, init, seed=seed, **kwargs)
        else:
            raise ValueError("Symmetric can only be used if reward input is depth.")

    def _render(self, mode="notebook", close=False, use_networkx=False):
        if close:
            return

        def color(val):
            if val > 0:
                return "#8EBF87"
            else:
                return "#F7BDC4"

        if use_networkx == True:
            if all("layout" in self.mdp_graph.nodes[node] for node in
                   self.mdp_graph.nodes(data=False)):
                layout = {node: self.mdp_graph.nodes[node]["layout"] for node in self.mdp_graph.nodes(data=False)}
            else:
                layout = None
            # label should either be revealed value, or empty if still unrevealed
            labels = {node: (str(self._state[node]) if not hasattr(self._state[node], 'sample') else '') for node in
                      self.mdp_graph.nodes(data=False)}
            node_color = [("#DEDEDE" if labels[node] == "" else color(float(labels[node]))) for node in range(len(self.mdp_graph.nodes(data=False)))]
            plot = nx.draw(self.mdp_graph, pos=layout, labels=labels, node_color=node_color, node_size=2000, arrowsize=30)
            return plot
        else:
            from graphviz import Digraph
    
            dot = Digraph()
            for x, ys in enumerate(self.tree):
                r = self._state[x]
                observed = not hasattr(self._state[x], "sample")
                c = color(r) if observed else "grey"
                label = str(round(r, 2)) if observed else str(x)
                dot.node(str(x), label=label, style="filled", color=c)
                for y in ys:
                    dot.edge(str(x), str(y))
            return dot

    def to_obs_tree(self, state, node, obs=(), sort=True):
        maybe_sort = sorted if sort else lambda x: x

        def rec(n):
            subjective_reward = state[n] if n in obs else expectation(state[n])
            children = tuple(maybe_sort(rec(c) for c in self.tree[n]))
            return (subjective_reward, children)

        return rec(node)


@lru_cache(SMALL_CACHE_SIZE)
def node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return smax(children, default=ZERO)


@lru_cache(CACHE_SIZE)
def exact_node_value_after_observe(obs_tree):
    """A distribution over the expected value of node, after making an observation.

    `obs` can be a single node, a list of nodes, or 'all'
    """
    children = tuple(exact_node_value_after_observe(c) + c[0] for c in obs_tree[1])
    return cmax(children, default=ZERO)

from mouselab.exact import hash_tree
from gym.spaces import Space
from mouselab.distributions import Categorical
from typing import Sequence, Union, Iterable
import numpy as np
class MultiCategorical(Space):
    def __init__(self, initial_states, initial_state_probabilities, unrevealed_sample_prob=0.50, seed=None):
        self.initial_states = initial_states
        self.initial_state_probabilities = initial_state_probabilities

        self.unrevealed_prob = unrevealed_sample_prob

        super().__init__(shape=[len(initial_states[0])], seed=seed)

    def sample(self):
        initial_state = self.np_random.choice(a=self.initial_states, p=self.initial_state_probabilities)
        unrevealed_node = self.np_random.choice(a=[True, False], p=[self.unrevealed_prob, 1 - self.unrevealed_prob],
                                                size=np.shape(initial_state))

        return [node if (unrevealed or not hasattr(node, "sample")) else node.sample() for node, unrevealed in
                zip(initial_state, unrevealed_node)]

    def contains(self, x) -> bool:
        """Does the space contain x?"""
        if isinstance(x, Sequence):
            x = np.array(x)  # done in MultiDiscrete class

        print(x)
        for node_idx, node in enumerate(x):
            if hasattr(node, "sample"):
                if node not in [initial_state[node_idx] for initial_state in self.initial_states]:
                    return False
            else:
                if node not in [val for initial_state in self.initial_states for val in (
                [initial_state[node_idx]] if not hasattr(initial_state[node_idx], "sample") else initial_state[
                    node_idx].vals)]:
                    return False
        return True
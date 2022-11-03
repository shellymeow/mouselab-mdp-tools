"""Agents that operate in discrete fully observable environments."""

import itertools as it
import time
from abc import ABC
from collections import defaultdict, deque
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from mouselab.utils import clear_screen

np.set_printoptions(precision=3, linewidth=200)


# ========================== #
# ========= Agents ========= #
# ========================== #


class RegistrationError(Exception):
    pass


class Agent(ABC):
    """An agent that can run openai gym environments."""

    def __init__(self):
        self.env = None
        self.policy = None
        self.ep_trace = None
        self.value_functions = []
        self.i_episode = 0
        self.memory = None

    def register(self, obj):
        """Attaches a component or env to this agent."""
        if isinstance(obj, list) and all([hasattr(item, "step") for item in obj]):
            self.env = obj  # list of envs with ground truth
        elif hasattr(obj, "step"):  # gym Env
            self.env = obj
        elif hasattr(obj, "act"):
            self.policy = obj
            obj.attach(self)
        elif hasattr(obj, "predict"):
            self.value_functions.append(obj)
            obj.attach(self)
        elif hasattr(obj, "batch"):
            self.memory = obj
        else:
            raise ValueError("Cannot register {}".format(obj))

    def run_episode(self, render=False, max_steps=1000, verbose=False):
        """Runs a single episode, returns a complete trace of the episode."""
        self.log = print if verbose else (lambda *args: None)

        if not self.env:
            raise RegistrationError("No environment registered.")
        if not self.policy:
            raise RegistrationError("No policy registered.")

        if isinstance(self.env, list):
            raise ValueError("Given list of trials please run agent.run_many().")
        else:
            trace = self._run_specific_episode(
                self.env, self.policy, max_steps=max_steps, render=render
            )
            return trace

    def _run_specific_episode(self, env, policy, max_steps=1000, render=False, fresh_start=True):
        # create trace dictionary
        trace = defaultdict(list)
        trace.update(
            {
                "i_episode": self.i_episode,
                "states": [],
                "actions": [],
                "rewards": [],
                "finished": False,
                "return": None,
            }
        )

        # start episode
        if fresh_start:
            new_state = env.reset()
        else:
            new_state = env._state
        self._start_episode(new_state)

        done = False
        num_steps = 0
        while not done and new_state != env.term_state:
            if num_steps == max_steps:
                raise BaseException(f"Reached max steps: {max_steps}")

            state = new_state

            if render:
                self._render(render)

            # get action for state
            action = policy.act(state)
            # take action for state
            new_state, reward, done, info = env.step(action)
            # experience this observation, for value functions
            self._experience(state, action, new_state, reward, done)

            # append experience to trace
            trace["states"].append(state)
            trace["actions"].append(action)
            trace["rewards"].append(reward)

            if done:
                trace["finished"] = True
                self._render(render)
                break

            num_steps += 1

        trace["states"].append(new_state)  # final state
        trace["return"] = sum(trace["rewards"])

        if self.memory is not None:
            self.memory.add(trace)
        self._finish_episode(trace)
        self.i_episode += 1
        return dict(trace)

    def run_many(self, num_episodes=None, pbar=True, fresh_start=True, **kwargs):
        """Runs several episodes, returns a summary of results."""
        if not self.env:
            raise RegistrationError("No environment registered.")
        if not self.policy:
            raise RegistrationError("No policy registered.")

        if num_episodes is None and not isinstance(self.env, list):
            raise ValueError("Either need list of envs as env or number of episodes")
        elif num_episodes is not None and isinstance(self.env, list):
            raise ValueError(
                "Either need list of envs as env or number of episodes (not both.)"
            )
        elif isinstance(self.env, list):
            num_episodes = len(self.env)

        # get the starting state if not fresh start
        if not fresh_start and not isinstance(self.env, list):
            starting_state = self.env._state

        data = defaultdict(list)
        for episode_idx in tqdm(range(num_episodes), disable=not pbar):
            if isinstance(self.env, list):
                trace = self._run_specific_episode(
                    self.env[episode_idx], self.policy, **kwargs
                )
            else:
                if fresh_start:
                    self.env.reset()
                else:
                    self.env._state = starting_state
                trace = self._run_specific_episode(self.env, self.policy, **kwargs)

            data["n_steps"].append(len(trace["states"]))
            for k, v in trace.items():
                data[k].append(v)

        return dict(data)

    def _start_episode(self, state):
        self.policy.start_episode(state)
        for vf in self.value_functions:
            vf.start_episode(state)

    def _finish_episode(self, trace):
        self.policy.finish_episode(trace)
        for vf in self.value_functions:
            vf.finish_episode(trace)

    def _experience(self, s0, a, s1, r, done):
        for vf in self.value_functions:
            vf.experience(s0, a, s1, r, done)

    def _render(self, mode):
        if mode == "step":
            x = input("> ")
            while x:
                print(eval(x))
                x = input("> ")
            clear_screen()
            self.env.render()
        elif mode == "clear":
            clear_screen()
            self.env.render()
        elif mode == "auto":
            time.sleep(0.4)
            clear_screen()
            self.env.render()
        elif mode == "notebook":
            self.env.render(mode=mode)
        else:
            pass


class Component(ABC):
    """A very abstract base class."""

    def __init__(self):
        super().__init__()
        self.agent = None
        self.saved = defaultdict(list)

    def experience(self, state, action, new_state, reward, done):
        """Learn from the results of taking action in state.

        state: state in which action was taken.
        action: action taken.
        new_state: the state reached after taking action.
        reward: reward received after taking action.
        done: True if the episode is complete.
        """
        pass

    def start_episode(self, state):
        """This function is run when an episode begins, starting at state."""
        pass

    def finish_episode(self, trace):
        """This function is run when an episode ends."""
        return

    def attach(self, agent):
        self.agent = agent

    @property
    def env(self):
        if isinstance(self.agent.env, list):
            return self.agent.env[self.agent.i_episode]
        else:
            return self.agent.env

    @property
    def i_episode(self):
        return self.agent.i_episode

    @property
    def observation_shape(self):
        if isinstance(self.agent.env, list):
            return self.env[self.i_episode].observation_space.shape
        else:
            return self.env.observation_space.shape

    @property
    def state_size(self):
        s = self.observation_space
        assert len(s) == 1
        return s[0]

    @property
    def n_action(self):
        if isinstance(self.env, list):
            return self.env[self.i_episode].action_space.n
        else:
            return self.env.action_space.n

    @property
    def memory(self):
        return self.agent.memory

    @property
    def ep_trace(self):
        return self.agent.ep_trace

    def log(self, *args):
        self.agent.log(*args)

    def save(self, key, val):
        self.saved[key].append(val)


class Memory(object):
    """Remembers past experiences."""

    def __init__(self, size=100000):
        self.states = deque(maxlen=size)
        self.actions = deque(maxlen=size)
        self.rewards = deque(maxlen=size)
        self.returns = deque(maxlen=size)

    def add(self, trace):
        self.states.extend(trace["states"])
        self.actions.extend(trace["actions"])
        self.actions.append(None)
        self.rewards.extend(trace["rewards"])
        self.rewards.append(0)
        self.returns.extend(np.flip(np.cumsum(np.flip(trace["rewards"], 0)), 0))
        self.returns.append(0)

        # self.experiences.extend(
        #     zip(
        #         trace["states"][:-1],
        #         trace["actions"],
        #         # trace['states'][1:],
        #         trace["rewards"],
        #         np.flip(np.cumsum(np.flip(trace["rewards"], 0)), 0),
        #     )
        # )
        # self.deque.append({"states": states, "rewards": rewards, "returns": returns})

    # def episodes(self, size, n=1):
    #     size = min(size, len(self.deque))
    #     if not self.deque:
    #         return
    #     for _ in range(n):
    #         yield np.random.choice(self.deque, size, replace=False)

    def batch(self, size):
        size = min(size, len(self.states))
        idx = np.random.choice(len(self.states), size=size, replace=False)
        return idx
        # return (self.experiences[i] for i in idx)


# class Memory(object):
#     """Remembers past experiences."""
#     Memory = namedtuple('Memory', ['states', 'rewards', 'returns'])
#     def __init__(self, size=100000):
#         self.episodes = deque(maxlen=size)
#         self.experiences
#         self.size = size

#     def add(self, trace):
#         # TODO this wastes RAM
#         states = np.stack(trace['states'])
#         returns = np.flip(np.cumsum(np.flip(trace['rewards'], 0)), 0)
#         rewards = np.array(trace['rewards'])
#         self.deque.append({'states': states, 'rewards': rewards, 'returns': returns})

#     def episodes(self, size, n=1):
#         size = min(size, len(self.deque))
#         if not self.deque:
#             return
#         for _ in range(n):
#             yield np.random.choice(self.deque, size, replace=False)


def run_episode(policy, env):
    agent = Agent()
    agent.register(env)
    agent.register(policy)
    return agent.run_episode()


def interactions(x):
    return [a * b for a, b in it.combinations(x, 2)]


class Model(object):
    """Simulated environment"""

    def __init__(self, env):
        self.env = deepcopy(env)

    def options(self, state):
        for a in range(self.env.action_space.n):
            self.env._state = state
            obs, r, done, info = self.env.step(a)
            yield a, self.env._state, r, done

import json
import time
from itertools import product

import dill as pickle
import numpy as np
from more_itertools import powerset

from mouselab.distributions import Categorical
from mouselab.mouselab import MouselabEnv


def get_possible_states_for_ground_truth(ground_truth, unrevealed_state):
    """
    Gets all possible revealed states for a given trial.
    :param ground_truth: Ground truth state as list (i.e. list of uncovered nodes)
    :param unrevealed_state:  Unrevealed state as list
                              (i.e. list of distribution objects for each node)
    :return: list of all possible states (all possible combinations of revealed nodes)
    """
    revealed_combinations = list(powerset(range(1, len(unrevealed_state) + 1)))
    # repeat ground truth for number of revealed combinations
    ground_truths = [np.asarray(ground_truth) for combination in revealed_combinations]

    # https://stackoverflow.com/a/28904614
    states = []
    for idx, (combination, curr_ground_truth) in enumerate(
        zip(revealed_combinations, ground_truths)
    ):
        curr_state = [
            unrevealed_state[state_idx] if state_idx in combination else entry
            for state_idx, entry in enumerate(curr_ground_truth)
        ]
        states.append(curr_state)

    return list(states)


def get_all_possible_ground_truths(categorical_gym_env):
    """
    Gets all possible ground truths for a MouselabEnv
    :param categorical_gym_env, instance of MouselabEnv
                with categorical or revealed states only
    :return: a list of all possible ground truths (i.e. values behind nodes)
    """
    possible_vals = [
        state.vals if isinstance(state, Categorical) else tuple([state])
        for state in categorical_gym_env._state
    ]

    possible_ground_truths = product(*possible_vals)
    return possible_ground_truths


def get_all_possible_states_for_ground_truths(categorical_gym_env, ground_truths):
    """
    Get all possible states for a list of ground truths
    (works with iterable of ground truths)
    """

    categorical_gym_env.reset()
    unrevealed_state = categorical_gym_env._state

    all_states = []
    for possible_ground_truth in ground_truths:
        states = get_possible_states_for_ground_truth(
            possible_ground_truth, unrevealed_state
        )
        all_states.extend(states)

    return all_states


def get_all_possible_states_for_env(categorical_gym_env):
    """
    Gets all possible states for a MouselabEnv
    :param categorical_gym_env, instance of MouselabEnv
                with categorical or revealed states only
    :return: a list of all possible states
                (i.e. for all ground truths, all possibly uncovered nodes)
    """
    possible_ground_truths = get_all_possible_ground_truths(categorical_gym_env)

    all_states = get_all_possible_states_for_ground_truths(
        categorical_gym_env, possible_ground_truths
    )

    return all_states


def deduplicate_states(complete_states, replacement_value=0, verbose=True):
    """
    Deduplicates states
    :param complete_states, a list of states
    :param replacement_value, a value that is not possible
                in any of the categorical distributions
    :param verbose whether to print out resulting size of deduplication
    :return: list of deduplicated states
    """
    complete_states = np.asarray(complete_states)
    # deduplicate states
    states_to_deduplicate = complete_states.copy()
    states_to_deduplicate[
        np.where(
            np.vectorize(lambda entry: isinstance(entry, Categorical))(
                states_to_deduplicate
            )
        )
    ] = replacement_value
    states, indices = np.unique(
        states_to_deduplicate.astype(np.float64), return_index=True, axis=0
    )

    if verbose:
        print(
            "{} states deduplicated, reduced to {}".format(
                complete_states.shape, states.shape
            )
        )

    return complete_states[indices, :]


def get_sa_pairs_from_states(states, terminal_action=None):
    """
    Gets all state action pairs from a list of states
    """
    all_sa_pairs = []
    for state in states:
        valid_actions = [
            idx
            for idx, list_item in enumerate(state)
            if isinstance(list_item, Categorical)
        ] + [len(state) if terminal_action is None else terminal_action]

        all_sa_pairs.extend([(tuple(state), action) for action in valid_actions])
    return all_sa_pairs


def get_all_possible_sa_pairs_for_env(
    categorical_gym_env, replacement_value=0, verbose=True
):
    """
    Gets all possible (state, action) pairs for a Categorical gym environment
    :param categorical_gym_env, instance of MouselabEnv
                with categorical or revealed states only
    :param replacement_value, a value that is not possible
                in any of the categorical distributions
    :param verbose whether to print out resulting size of deduplication
    :return: list of all state, action pairs as tuples
    """
    all_states = get_all_possible_states_for_env(categorical_gym_env)
    dedup_states = deduplicate_states(
        all_states, replacement_value=replacement_value, verbose=verbose
    )

    all_sa_pairs = get_sa_pairs_from_states(dedup_states, terminal_action=categorical_gym_env.term_action)

    if categorical_gym_env.include_last_action:
        return add_extended_state_to_sa_pairs(all_sa_pairs)
    else:
        return all_sa_pairs


def add_extended_state_to_sa_pairs(all_sa_pairs):
    all_sa_pairs_with_last_action = []
    for s, a in all_sa_pairs:
        if all(hasattr(action, "sample") for action_index, action in enumerate(s[1:-1])):
            all_sa_pairs_with_last_action.append((s, a))
        else:
            all_sa_pairs_with_last_action.extend(
                [((*s[:-1], action_index + 1), a) for action_index, action in enumerate(s[1:-1]) if
                 not hasattr(action, "sample")])
    return all_sa_pairs_with_last_action

def add_extended_state_to_state_list(all_states):
    all_states_with_last_action = []
    for s in all_states:
        if all(hasattr(action, "sample") for action_index, action in enumerate(s[1:-1])):
            all_states_with_last_action.append((s, a))
        else:
            all_states_with_last_action.extend(
                [((*s[:-1], action_index + 1), a) for action_index, action in enumerate(s[1:-1]) if
                 not hasattr(action, "sample")])
    return all_sa_pairs_with_last_action

# ========================================================================================
#
# These functions are used to extract states from a JSON file of possible trials
#
# ========================================================================================


def save_all_states(
    complete_states, save_location, extra_info="", replacement_value=0, verbose=True
):
    """
    Saves all states
    :param complete_states: An array-like object with mouselab environment states
    :param save_location:A location to save the states (using pathlib)
    :param extra_info: Extra info to put in the file name of the saved states
    :param replacement_value: Value to replace any distribution objects
                with during deduplication
            Warning: this should not be a possible value of uncovered states!!!
    :return: complete, deduplicated states
    """
    complete_states = deduplicate_states(
        complete_states, replacement_value=replacement_value, verbose=verbose
    )

    # save states
    if save_location is not None:
        file = open(
            save_location.joinpath(
                f"complete_states{extra_info}_{time.strftime('%Y%m%d-%H%M')}.pickle"
            ),
            "wb",
        )
        pickle.dump(complete_states, file)
        file.close()
    return complete_states


def get_ground_truths_from_json(ground_truth_file):
    """
    gets ground truth states from json
    """
    # open ground truth file
    ground_truth_file = json.load(open(ground_truth_file, "rb"))

    # get ground truth stateRewards
    ground_truths = [entry["stateRewards"] for entry in ground_truth_file]

    return ground_truths


def get_states_from_json(ground_truth_file, experiment_setting="high_increasing"):
    """
    Gets states from input JSON file
                (what we use to generate experiment trials in experiments)
    :param ground_truth_file: full path to ground truth file
    :param experiment_setting: Name of experiment setting
            WARNING: assumes trials all have same experiment setting
    :return: all possible states given a ground truth setting
    """
    ground_truths = get_ground_truths_from_json(ground_truth_file)

    # used to extract unrevealed state (usually a list of distributions)
    unrevealed_state = MouselabEnv.new_symmetric_registered(experiment_setting).init

    # gets all combinations of possibly revealed states for each ground truth setting
    possible_states = np.vstack(
        [
            get_possible_states_for_ground_truth(ground_truth, unrevealed_state)
            for ground_truth in ground_truths
        ]
    )

    # save all states
    save_all_states(possible_states)

    return possible_states


def get_num_actions(branching):
    """
    branching: list denoting branching of mouselab experiment
    """
    actions = 0
    for depth in range(len(branching)):
        actions += np.prod(branching[: depth + 1])

    # add final action
    actions += 1
    return actions

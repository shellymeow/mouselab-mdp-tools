from contexttimer import Timer

from mouselab.env_utils import (
    get_all_possible_sa_pairs_for_env,
    get_all_possible_states_for_ground_truths,
    get_sa_pairs_from_states,
)
from mouselab.exact import solve, backward_solve, hash_tree
from mouselab.env_utils import add_extended_state_to_sa_pairs

from toolz import unique

def timed_solve_env(
    env, verbose=True, save_q=False, ground_truths=None, backwards=False, hash_key=None, dedup_by_hash=False, **solve_kwargs
):
    """
    Solves environment, saves elapsed time and optionally prints value and elapsed time
    :param env: MouselabEnv with only discrete distribution (must not be too big)
    :param verbose: Whether or not to print out solve information once done
    :param save_q:
    :param ground_truths:
    :param backwards:
    :param hash_key:
    :param solve_kwargs:
    :return: Q, V, pi, info
             Q, V, pi are all recursive functions
             info contains the number of times Q and V were called
                as well as the elapsed time ("time")
    """
    if backwards:
        if hash_key is None:
            hash_key = lambda sa_pair: hash_tree(env, *sa_pair)
        with Timer() as t:
            info = {}
            Q = backward_solve(env, **solve_kwargs)
            info["time"] = t.elapsed
            if verbose:
                optimal_value = sum(
                    p * max(Q[hash_key((s,a))] for a in env.actions(s))
                    for s, p in zip(env.initial_states, env.initial_state_probabilities)
                )
                print("optimal -> {:.2f} in {:.3f} sec".format(optimal_value, t.elapsed))

            if save_q:
                info["q_dictionary"] = construct_q_dictionary(Q, env, ground_truths=ground_truths, verbose=verbose, hash_key=hash_key, dedup_by_hash=dedup_by_hash)

        return None, None, None, info
    else:
        with Timer() as t:
            Q, V, pi, info = solve(env, **solve_kwargs)
            info["time"] = t.elapsed
            if verbose:
                optimal_value = sum(
                    p * V(s)
                    for s, p in zip(env.initial_states, env.initial_state_probabilities)
                )
                print("optimal -> {:.2f} in {:.3f} sec".format(optimal_value, t.elapsed))
            elif save_q:
                # call V to cache q_dictionary
                for s in env.initial_states:
                    V(s)

            #  Save Q function
            if save_q:
                if dedup_by_hash:
                    hash_key = lambda sa_pair: hash_tree(env, *sa_pair)
                # In some cases, it is too costly to save whole Q function
                info["q_dictionary"] = construct_q_dictionary(Q, env, ground_truths=ground_truths, verbose=verbose, hash_key=hash_key, dedup_by_hash=dedup_by_hash)

        return Q, V, pi, info


def construct_q_dictionary(Q, env, ground_truths=None, verbose=False, hash_key=None, dedup_by_hash=True):
    """
    Construct q dictionary for env, given environment is solved
    """
    if ground_truths is None:
        sa = get_all_possible_sa_pairs_for_env(env, verbose=verbose)
    else:
        all_possible_states = get_all_possible_states_for_ground_truths(
            env, ground_truths
        )
        sa = get_sa_pairs_from_states(all_possible_states)

        if env.include_last_action:
            sa = add_extended_state_to_sa_pairs(sa)

    if dedup_by_hash and hash_key:
        sa = unique(sa, key=lambda sa_pair: hash_key(sa_pair))

    if isinstance(Q, dict):
        if hash_key is None:
            hash_key = lambda sa_pair: sa_pair

        q_dictionary = {pair: Q[hash_key(pair)] for pair in sa}
    else:
        q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary

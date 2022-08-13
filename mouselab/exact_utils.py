from contexttimer import Timer

from mouselab.env_utils import (
    get_all_possible_sa_pairs_for_env,
    get_all_possible_states_for_ground_truths,
    get_sa_pairs_from_states,
)
from mouselab.exact import solve
from mouselab.env_utils import add_extended_state_to_sa_pairs

def timed_solve_env(
    env, verbose=True, save_q=False, ground_truths=None, **solve_kwargs
):
    """
    Solves environment, saves elapsed time and optionally prints value and elapsed time
    :param env: MouselabEnv with only discrete distribution (must not be too big)
    :param verbose: Whether or not to print out solve information once done
    :return: Q, V, pi, info
             Q, V, pi are all recursive functions
             info contains the number of times Q and V were called
                as well as the elapsed time ("time")
    """
    with Timer() as t:
        Q, V, pi, info = solve(env, **solve_kwargs)
        info["time"] = t.elapsed
        if verbose:
            optimal_value = sum(
                V(s)
                for s, p in zip(env.initial_states, env.initial_state_probabilities)
            )
            print("optimal -> {:.2f} in {:.3f} sec".format(optimal_value, t.elapsed))
        elif save_q:
            # call V to cache q_dictionary
            for s in env.initial_states:
                V(s)

        #  Save Q function
        if save_q is not None and ground_truths is not None:
            if env.include_last_action:
                # need to modify ground truths to include last clicked state
                ground_truths = add_extended_state_to_sa_pairs(ground_truths)
            # In some cases, it is too costly to save whole Q function
            info["q_dictionary"] = construct_partial_q_dictionary(Q, env, ground_truths)
        elif save_q is not None:
            info["q_dictionary"] = construct_q_dictionary(Q, env, verbose)

    return Q, V, pi, info


def construct_q_dictionary(Q, env, verbose=False):
    """
    Construct q dictionary for env, given environment is solved
    """

    sa = get_all_possible_sa_pairs_for_env(env, verbose=verbose)
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary


def construct_partial_q_dictionary(Q, env, selected_ground_truths):
    """
    Construct q dictionary for only specified ground truth values
    """
    all_possible_states = get_all_possible_states_for_ground_truths(
        env, selected_ground_truths
    )
    sa = get_sa_pairs_from_states(all_possible_states)
    
    if env.include_last_action:
        sa = add_extended_state_to_sa_pairs(sa)
        
    q_dictionary = {pair: Q(*pair) for pair in sa}
    return q_dictionary

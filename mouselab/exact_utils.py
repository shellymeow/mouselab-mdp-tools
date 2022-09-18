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
    :param save_q:
    :param ground_truths:
    :param solve_kwargs:
    :return: Q, V, pi, info
             Q, V, pi are all recursive functions
             info contains the number of times Q and V were called
                as well as the elapsed time ("time")
    """
    if save_q is True:
        q_dictionary={}
    else:
        q_dictionary=None
        
    with Timer() as t:
        Q, V, pi, info = solve(env, q_dictionary=q_dictionary,**solve_kwargs)
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
        if save_q is not None:
            if ground_truths is not None:
                # In some cases, it is too costly to save whole Q function
                all_possible_states = get_all_possible_states_for_ground_truths(
                    env, ground_truths
                )

                sa_pairs = get_sa_pairs_from_states(all_possible_states)

                if env.include_last_action:
                    sa_pairs = add_extended_state_to_sa_pairs(sa_pairs)
            else:
                sa_pairs = get_all_possible_sa_pairs_for_env(env, verbose=verbose)

            sa_pairs = sorted(sa_pairs,
                              key=lambda sa_pair: len([node for node in sa_pair[0] if hasattr(node, 'sample')]))

            if len(set(q_dictionary.keys()).intersection(set(sa_pairs))) < len(set(sa_pairs)):
                [Q(*sa_pair) for sa_pair in sa_pairs if sa_pair not in set(q_dictionary.keys())]

            info["q_dictionary"] = {key: q_dictionary[key] for key in set(sa_pairs) & set(q_dictionary.keys())}

            info["q_dictionary"] = q_dictionary

    return Q, V, pi, info


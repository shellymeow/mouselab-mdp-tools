from toolz import memoize, unique
from mouselab.env_utils import get_all_possible_ground_truths

def sort_tree(env, state):
    """Breaks symmetry between belief states.

    This is done by enforcing that the knowledge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    state = list(state)
    for i in range(len(env.tree) - 1, -1, -1):
        if not env.tree[i]:
            continue
        c1, c2 = env.tree[i]
        idx1, idx2 = env.subtree_slices[c1], env.subtree_slices[c2]

        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)


def hash_tree(env, state, action=None, last_action_info=None):
    """Breaks symmetry between belief states."""
    if state == "__term_state__":
        return hash(state)

    if env.include_last_action and action is not env.term_action:
        if last_action_info is None:
            node_last_clicks = [0 for action_idx, action in enumerate(state[:-1])]
            node_last_clicks[state[-1]] = 1
        else:
            node_last_clicks = last_action_info[state[-1]]
        state = [*zip(state[:-1],node_last_clicks)]

    if action is not None and action is not env.term_action:
        actions = [0 for _ in state]
        actions[action] = 1
        if env.include_last_action:
            state = [tuple([*curr_state, action]) for curr_state, action in zip(state, actions)]
        else:
            state = [*zip(state, actions)]

    def rec(n):
        x = hash(state[n])
        childs = sum(rec(c) for c in env.tree[n])
        return hash(str(x + childs))

    return rec(0)

def solve(env, hash_state=None, actions=None, blinkered=None, last_action_info=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    info = {"q": 0, "v": 0}  # track number of times each function is called

    if hash_state is None:
        if hasattr(env, "n_arm"):
            hash_state = lambda state: tuple(sorted(state))
        elif hasattr(env, "tree"):
            # hash_state = lambda state: sort_tree(env, state)
            hash_state = lambda state: hash_tree(env, state, last_action_info=last_action_info)

    # for tests with no hashing
    if hash_state == "test":
        hash_state = None

    if actions is None:
        actions = env.actions
    if blinkered == "recursive":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], *env.path_to(a)[:-1], env.term_action)

    elif blinkered == "children":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], env.term_action)

    elif blinkered == "branch":
        assert hasattr(env, "_relevant_subtree")

        def subset_actions(a):
            if a == env.term_action:
                return ()
            else:
                return (*env._relevant_subtree(a), env.term_action)

    elif blinkered:

        def subset_actions(a):
            return (a, env.term_action)

    else:
        subset_actions = lambda a: None

    if hash_state is not None:

        def hash_key(args, kwargs):
            state = args[0]
            if state is None:
                return state
            else:
                if kwargs:
                    # Blinkered approximation. Hash key is insensitive
                    # to states that can't be acted on, except for the
                    # best expected value
                    # Embed the action subset into the state.
                    action_subset = kwargs["action_subset"]
                    mask = [0] * len(state)
                    for a in action_subset:
                        mask[a] = 1
                    state = tuple(zip(state, mask))
                return hash_state(state)

    else:
        hash_key = None

    @memoize
    def Q(s, a):
        info["q"] += 1
        action_subset = subset_actions(a)
        return sum(p * (r + V(s1, action_subset)) for p, s1, r in env.results(s, a))

    @memoize(key=hash_key)
    def V(s, action_subset=None):
        if s is None:
            return 0
        info["v"] += 1
        acts = actions(s)
        if action_subset is not None:
            acts = tuple(a for a in acts if a in action_subset)
        return max((Q(s, a) for a in acts), default=0)

    @memoize
    def pi(s):
        return max(actions(s), key=lambda a: Q(s, a))

    return Q, V, pi, info

def get_previous_sa_pairs(env, sa_pair):
    if env.include_last_action:
        state = sa_pair[0][:-1]
    else:
        state = sa_pair[0]

    previous_sa_pairs = []
    for initial_state in env.initial_states:
        for node_idx, node in enumerate(initial_state):
            if hasattr(node, 'sample'):
                curr_state = list(state)
                curr_state[node_idx] = node

                if env.include_last_action:
                    previous_sa_pairs.extend([((*curr_state, last_node_idx),node_idx) for last_node_idx, last_node in enumerate(curr_state) if not hasattr(last_node, 'sample') and last_node_idx != 0])
                    previous_sa_pairs.extend(
                        [((*curr_state, last_node_idx), env.term_action) for last_node_idx, last_node in enumerate(curr_state) if
                            not hasattr(last_node, 'sample') and last_node_idx != 0])

                else:
                    previous_sa_pairs.extend(
                        [(tuple(curr_state), node_idx)])
                    previous_sa_pairs.extend(
                        [(tuple(curr_state), env.term_action)])
    return previous_sa_pairs

def backward_solve(env, hash_key=None, verbose=True, last_action_info=None):
    """
    Rather than using memoization, we go backwards through the (state, action) pairs:
        First through terminal action
        Then through states from most revealed to the least revealed
    """
    if hash_key is None:
        if hasattr(env, "tree"):
            hash_key = lambda sa_pair: hash_tree(env, *sa_pair, last_action_info=last_action_info)

    # for tests with no hashing
    if hash_key == "test":
        hash_key = lambda sa_pair: sa_pair

    if env.include_last_action:
        sa_pairs = [((*ground_truth, 0), env.term_action) for ground_truth in get_all_possible_ground_truths(env)]
    else:
        sa_pairs = [(ground_truth, env.term_action) for ground_truth in get_all_possible_ground_truths(env)]

    sa_pairs = list(unique(sa_pairs, key=lambda sa_pair: hash_key(sa_pair)))
    Q = {hash_key(sa_pair): None for sa_pair in sa_pairs}

    sa_pair_idx = 0
    while len(sa_pairs) > 0:
        sa_pair = sa_pairs.pop(0)

        if sa_pair_idx % 50000 == 0 and verbose:
            print(sa_pair_idx)
        Q[hash_key(sa_pair)] = sum(transition_probability * (reward + max((Q[hash_key((next_state, action))] for action in env.actions(next_state)), default=0)) for transition_probability, next_state, reward in env.results(*sa_pair))

        sa_pair_idx += 1

        new_sa_pairs = get_previous_sa_pairs(env, sa_pair)
        new_sa_pairs = list(unique(new_sa_pairs, key=lambda sa_pair: hash_key(sa_pair)))
        new_sa_pairs = [sa_pair for sa_pair in new_sa_pairs if hash_key(sa_pair) not in Q.keys()]

        sa_pairs.extend(new_sa_pairs)
        Q = {**Q, **{hash_key(sa_pair): None for sa_pair in new_sa_pairs}}

    return Q
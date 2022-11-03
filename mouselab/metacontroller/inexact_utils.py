from contexttimer import Timer

from mouselab.env_utils import get_all_possible_states_for_ground_truths, get_sa_pairs_from_states, add_extended_state_to_sa_pairs
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
import numpy as np
from mouselab.agents import Agent
from mouselab.policies import LiederPolicy
from toolz import unique

def timed_solve_env(
    env, rollout_function, verbose=True, ground_truths=None, hash_key=None, dedup_by_hash=False
):
    with Timer() as timer:
        states = get_all_possible_states_for_ground_truths(
            categorical_gym_env=env, ground_truths=ground_truths)
        sa_pairs = get_sa_pairs_from_states(states)
        if env.include_last_action:
            sa_pairs = add_extended_state_to_sa_pairs(sa_pairs)

        if verbose:
            print(f"Got (s,a) pairs, time elapsed: {timer.elapsed}")

    if verbose:
        print(f"Deduping (s,a) to save, time elapsed: {timer.elapsed}")
    if dedup_by_hash and hash_key:
        sa_pairs = unique(sa_pairs, key=lambda sa_pair: hash_key(env, *sa_pair))
    else:
        sa_pairs = unique(sa_pairs)

    Q = {}
    for sa_pair in sa_pairs:
        Q[sa_pair] = rollout_function(sa_pair)

    return Q

def ground_truth_compatible_with_state(state, ground_truth):
    return all([ground_truth_node == state_node for state_node, ground_truth_node in zip(state, ground_truth) if
            not hasattr(state_node, 'sample')])    
        
def get_rollouts_for_ground_truths(weights,
                                   state,
                                   action,
                                   tree,
                                   init,
                                   ground_truths,
                                   cost,
                                   features,
                                   env_params,
                                   mdp_graph_properties=None,
                                   num_repetitions=2):
    

    envs = [MetaControllerMouselab(tree, init, cost=cost, term_belief=False, features=features, seed=91, mdp_graph_properties=mdp_graph_properties,
                                   ground_truth=ground_truth, **env_params) for ground_truth in ground_truths if ground_truth_compatible_with_state(state, ground_truth)] * num_repetitions

    rewards = [0 for _ in envs]
    for env_num, curr_env in enumerate(envs):
        curr_env._state = state
        next_state, rewards[env_num], done, _ = curr_env.step(action)
        curr_env._state = next_state

    agent = Agent()

    agent.register(envs)
    agent.register(LiederPolicy(theta=weights.flatten()))

    trace = agent.run_many(pbar=False, fresh_start=False)

    possible_values = [curr_reward + sum(curr_trace_rewards) for curr_reward, curr_trace_rewards in
                       zip(rewards, trace["rewards"])]
    return np.mean(possible_values)

if __name__ == "__main__":
    from mouselab.metacontroller.vanilla_BMPS import load_feature_file
    from mouselab.mouselab import MouselabEnv
    from mouselab.cost_functions import linear_depth

    from mouselab.distributions import Categorical
    from mouselab.envs.registry import register

    optimization_kwargs, features, _, secondary_variables = load_feature_file("Basic")

    register(
        name="small_test_case",
        branching=[1, 2],
        reward_inputs="depth",
        reward_dictionary={1: Categorical([-500]), 2: Categorical([-60, 60])},
    )

    env = MouselabEnv.new_symmetric_registered("small_test_case")
    cost = linear_depth(depth_cost_weight=5.0, static_cost_weight=2.0)
    ground_truths = [[0, -500, -60, -60], [0, -500, 60, 60]]
    env = MetaControllerMouselab(env.tree, env.init, cost=cost, term_belief=False, features=features, seed=91)

    W_vanilla = np.array([[0.5,0.2,2,0.3]])
    rollout_function = lambda sa_pair : get_rollouts_for_ground_truths(W_vanilla, sa_pair[0], sa_pair[1], env.tree, env.init, ground_truths, num_repetitions=5)

    print(env.init)
    from mouselab.exact import hash_tree
    q = timed_solve_env(
            env, rollout_function, verbose=True, ground_truths=ground_truths, hash_key=hash_tree, dedup_by_hash=True
    )
    print(q)
    print(len(q))

    q = timed_solve_env(
            env, rollout_function, verbose=True, ground_truths=ground_truths, hash_key=None, dedup_by_hash=False
    )
    print(q)
    print(len(q))
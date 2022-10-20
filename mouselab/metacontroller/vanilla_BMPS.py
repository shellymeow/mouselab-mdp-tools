"""
Adapted from: https://github.com/RationalityEnhancement/SSD_Hierarchical/tree/master/Metacontroller
"""
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
import numpy as np
import GPyOpt
import time

def original_mouselab(W, tree, init, cost ,num_episodes=100, SEED=1000, term_belief=False, exact_seed=False, cost_function="Basic"):
    """[summary]

    Args:
        W ([type]): [description]
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of clicking on a node.
        num_episodes (int, optional): Number of episodes to evaluate the MDP on. Defaults to 100.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """

    w1 = W[:,0]
    w2 = W[:,1]
    w4 = W[:,2]
    if cost_function == "Actionweight" or cost_function == "Independentweight":
        w0 = W[:,3]
    w3 = 1 - w1 - w2

    if cost_function == "Basic":
        simple_cost = True
    else:
        simple_cost = False

    num_nodes = len(tree) - 1

    def voc_estimate(x):
        features = env.action_features(x)
        if cost_function == "Basic":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
        elif cost_function == "Hierarchical":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
        elif cost_function == "Actionweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
        elif cost_function == "Novpi":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/num_nodes))
        elif cost_function == "Proportional":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/num_nodes))
        elif cost_function == "Independentweight":
            return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
        else:
            assert False

    rewards = []
    actions = []
    for i in range(num_episodes):
        if exact_seed:
            np.random.seed(SEED + i)
        else:
            np.random.seed(1000*SEED + i)
        env = MetaControllerMouselab(tree, init, cost=cost, term_belief=term_belief, simple_cost=simple_cost)
        exp_return = 0
        actions_tmp = []
        while True:
            possible_actions = list(env.actions(env._state))
            action_taken = max(possible_actions, key = voc_estimate)
            _, rew, done, _=env._step(action_taken)
            exp_return+=rew
            if done:
                break
            else:
                actions_tmp.append(action_taken)
        rewards.append(exp_return)
        actions.append(actions_tmp)
        del env, possible_actions
    return rewards, actions


def optimize(tree, init, cost, evaluated_episodes=100, samples=30, iterations=50, SEED=0, term_belief=True, exact_seed=False, cost_function="Basic", verbose=False):
    """Optimizes the weights for BMPS using Bayesian optimization.

    Args:
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of computing a node.
        samples (int, optional): Number of initial random guesses before optimization. Defaults to 30.
        iterations (int, optional): Number of optimization steps. Defaults to 50.
        SEED (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to True.
    """

    def blackbox_original_mouselab(W):
        w1 = W[:,0]
        w2 = W[:,1]
        w4 = W[:,2]
        if cost_function == "Actionweight" or cost_function == "Independentweight":
            w0 = W[:,3]
        w3 = 1 - w1 - w2

        if cost_function == "Basic":
            simple_cost = True
        else:
            simple_cost = False

        num_episodes = evaluated_episodes
        if verbose:
            print("Weights", W)

        num_nodes = len(tree) - 1

        def voc_estimate(x):
            features = env.action_features(x)
            if cost_function == "Basic":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0]
            elif cost_function == "Hierarchical":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*features[0][1] + w2*features[0][2])
            elif cost_function == "Actionweight":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + w0*(features[0][1]))
            elif cost_function == "Novpi":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(w1*features[0][0] + w3*(features[0][1]/num_nodes))
            elif cost_function == "Proportional":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*(features[0][0] + (features[0][1]/num_nodes))
            elif cost_function == "Independentweight":
                return w1*features[1] + w2*features[3] + w3*features[2] + w4*features[0][0] + w0*features[0][1]
            else:
                assert False

        cumreturn = 0
        for i in range(num_episodes):
            #TODO
            if exact_seed:
                np.random.seed(SEED + i)
            else:
                np.random.seed(1000*SEED + i)
            env = MetaControllerMouselab(tree, init, cost=cost, term_belief=term_belief, simple_cost=simple_cost)
            exp_return = 0
            actions_tmp = []
            while True:
                possible_actions = list(env.actions(env._state))
                action_taken = max(possible_actions, key = voc_estimate)
                _, rew, done, _=env._step(action_taken)
                exp_return+=rew
                if done:
                    break
            cumreturn += exp_return
            del env
        if verbose:
            print("Return", cumreturn/num_episodes)
        return - (cumreturn/num_episodes)

    if verbose:
        print(cost_function)
    np.random.seed(123456)

    if cost_function == "Actionweight" or cost_function == "Independentweight":
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w4', 'type': 'continuous', 'domain': (1,len(tree)-1)},
            {'name': 'w0', 'type': 'continuous', 'domain': (0,1)}]
    else:
        space = [{'name': 'w1', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w2', 'type': 'continuous', 'domain': (0,1)},
            {'name': 'w4', 'type': 'continuous', 'domain': (1,len(tree)-1)}]

    constraints = [{'name': 'part_1', 'constraint': 'x[:,0] + x[:,1] - 1'}]

    feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)

    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, samples)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(blackbox_original_mouselab)

    # --- CHOOSE the model type
    #This model does Maximum likelihood estimation of the hyper-parameters.
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=False)

    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    bo = GPyOpt.methods.ModularBayesianOptimization(model, feasible_region, objective, acquisition, evaluator, initial_design)

    # --- Stop conditions
    max_time  = None
    tolerance = 1e-6     # distance between two consecutive observations

    # Run the optimization
    max_iter  = iterations
    time_start = time.time()
    train_tic = time_start
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=True)

    W_low = np.array([bo.x_opt])
    train_toc = time.time()

    if verbose:
        print("\nSeconds:", train_toc-train_tic)
        print("Weights:", W_low)
    blackbox_original_mouselab(W_low)

    return W_low, train_toc-train_tic

def eval(W_low, n, tree, init, cost, SEED=1000, term_belief=False, verbose=True, exact_seed=False, cost_function="Basic"):
    """Evaluates the BMPS weights and logs the execution time.

    Args:
        W_low (np.array): BMPS weights
        n (int): Number of episodes for evaluation.
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of computing a non goal node.
        SEED (int, optional): Seed to fix random MDP initialization. Defaults to 1000.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """

    eval_tic = time.time()
    rewards, actions = original_mouselab(W=W_low, tree=tree, init=init, cost=cost, SEED=SEED, num_episodes=n, term_belief=term_belief, exact_seed=exact_seed, cost_function=cost_function)
    if verbose:
        print("Seconds:", time.time() - eval_tic)
        print("Average reward:", np.mean(rewards))
    return rewards, actions

if __name__ == "__main__":
    BO_RESTARTS = 10
    BO_STEPS = 100
    EVAL_EPISODES = 100

    from mouselab.mouselab import MouselabEnv
    from mouselab.cost_functions import linear_depth
    env = MouselabEnv.new_symmetric_registered("high_increasing")
    cost = linear_depth(depth_cost_weight=5.0, static_cost_weight=2.0)

    W_vanilla, time_vanilla = optimize(tree=env.tree, init=env.init, cost=cost, SEED=91, samples=BO_RESTARTS,
                                       iterations=BO_STEPS, evaluated_episodes=EVAL_EPISODES, cost_function="Basic", verbose=False)
    rewards_vanilla, actions_vanilla = eval(W_vanilla, n=500, tree=env.tree, init=env.init, cost=cost, SEED=91,
                                            cost_function="Basic", exact_seed=True)
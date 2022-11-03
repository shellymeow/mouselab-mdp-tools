"""
Adapted from: https://github.com/RationalityEnhancement/SSD_Hierarchical/tree/master/Metacontroller
"""
from mouselab.metacontroller.mouselab_env import MetaControllerMouselab
import numpy as np
import GPyOpt
import time
from numpy.random import default_rng
from mouselab.policies import LiederPolicy
from mouselab.agents import Agent
from pathlib import Path

import yaml

from mouselab.mouselab import MouselabEnv
from mouselab.cost_functions import linear_depth

def add_secondary_variables(W, secondary_variables):
    return np.append(W, [eval(secondary_variables[secondary_variable])(W) for secondary_variable in secondary_variables])

def original_mouselab(W, tree, init, cost ,num_episodes=100, seed=None, term_belief=False, features=None, verbose=False, **env_kwargs):
    """[summary]

    Args:
        W  (np.array, iterable): BMPS weights
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of clicking on a node.
        num_episodes (int, optional): Number of episodes to evaluate the MDP on. Defaults to 100.
        seed (int, optional): Seed to fix random MDP initialization.
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to False.
        features (list, str):

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """
    agent = Agent()

    env = [MetaControllerMouselab(tree, init, cost=cost, term_belief=term_belief, features=features, seed=seed, **env_kwargs) for _ in range(num_episodes)]
    agent.register(env)

    agent.register(LiederPolicy(theta=W.flatten()))

    trace = agent.run_many(pbar=verbose)

    return trace["return"], trace["actions"]


def optimize_bmps_weights(tree, init, cost, num_episodes=100, samples=30, iterations=50, seed=None, optimization_seed=123456, term_belief=True, features=None, optimization_kwargs=None, secondary_variables=None,
             verbose=False, **env_kwargs):
    """Optimizes the weights for BMPS using Bayesian optimization.

    Args:
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of computing a node.
        samples (int, optional): Number of initial random guesses before optimization. Defaults to 30.
        iterations (int, optional): Number of optimization steps. Defaults to 50.
        seed (int, optional): Seed to fix random MDP initialization.
        optimization_seed (int, optional): Seed for optimization
        term_belief (bool, optional): If true the expected reward instead of the real reward is used. Defaults to True.
    """

    def blackbox_original_mouselab(W):
        W = add_secondary_variables(W, secondary_variables=secondary_variables)

        returns, actions = original_mouselab(W, tree, init, cost, num_episodes=num_episodes, seed=seed, term_belief=term_belief, features=features,
                          verbose=verbose, **env_kwargs)

        if verbose:
            print("Return", np.sum(returns) / num_episodes)

        return - np.sum(returns) / num_episodes

    # not best practice, but seems this optimization library is past support:
    # https://github.com/SheffieldML/GPyOpt/issues/337
    np.random.seed(optimization_seed)

    optimization_kwargs["space"] = [{key: (eval(val) if key == "domain" else val) for key, val in
                                     curr_param.items()} for curr_param in optimization_kwargs["space"] if curr_param["type"] != "secondary"]
    feasible_region = GPyOpt.Design_space(**optimization_kwargs)

    initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, samples)
    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(blackbox_original_mouselab)

    # --- CHOOSE the model type
    #This model does Maximum likelihood estimation of the hyper-parameters.
    model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=10,verbose=verbose)

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
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbosity=verbose)

    W_low = np.array([bo.x_opt])
    train_toc = time.time()

    if verbose:
        print("\nSeconds:", train_toc-train_tic)
        print("Weights:", W_low)

    blackbox_original_mouselab(W_low)
    W_low = add_secondary_variables(W_low, secondary_variables=secondary_variables)

    return W_low, train_toc-train_tic

def eval_bmps_weights(W, num_episodes, tree, init, cost, seed=None, term_belief=False, verbose=True, features=None):
    """Evaluates the BMPS weights and logs the execution time.

    Args:
        W (np.array): BMPS weights
        num_episodes (int): Number of episodes for evaluation.
        tree ([int]): MDP structure
        init ([int]): MDP reward distribution per node
        cost (callable): Cost of computing a non goal node.
        seed (int, optional): Seed to fix random MDP initialization. Defaults to 1000.

    Returns:
        [int]: Reward of each episode
        [[int]]: Actions of each episode
    """

    if verbose:
        eval_tic = time.time()
    rewards, actions = original_mouselab(W=W, tree=tree, init=init, cost=cost, seed=seed, num_episodes=num_episodes, term_belief=term_belief, features=features, verbose=verbose)
    if verbose:
        print("Seconds:", time.time() - eval_tic)
        print("Average reward:", np.mean(rewards))
    return rewards, actions

def load_feature_file(filename, path=None):
    if path is None:
        path = Path(__file__).parents[0] / "bmps_features"

    with open(path / f"{filename}.yaml", "rb") as f:
        feature_inputs = yaml.safe_load(f)

    optimization_kwargs = {
        "space" : [{"name": feature, **details} for feature, details in feature_inputs["features"].items()],
        "constraints": [{"name": feature, **details} for feature, details in feature_inputs["constraints"].items()]}
    # add any additional arguments, could store optimization hyperparameters in YAML file
    additional_kwargs = {key : val for key, val in feature_inputs.items() if key not in ["features", "constraints", "initial_function"]}
    features = list(feature_inputs["features"].keys())

    return optimization_kwargs, features, additional_kwargs, feature_inputs["secondary_variables"]


if __name__ == "__main__":
    BO_RESTARTS = 10
    BO_STEPS = 50
    EVAL_EPISODES = 100

    env = MouselabEnv.new_symmetric_registered("high_increasing")
    cost = linear_depth(depth_cost_weight=5.0, static_cost_weight=2.0)

    for feature_file in ["Basic"]:
        print(feature_file)
        optimization_kwargs, features, _, secondary_variables = load_feature_file(feature_file)

        W_vanilla, time_vanilla = optimize_bmps_weights(tree=env.tree, init=env.init, cost=cost, seed=91, samples=BO_RESTARTS,
                                           iterations=BO_STEPS, num_episodes=EVAL_EPISODES, features=features, optimization_kwargs=optimization_kwargs, secondary_variables=secondary_variables, verbose=False)
        print(W_vanilla, time_vanilla)
        actions_vanilla = eval_bmps_weights(W_vanilla, num_episodes=EVAL_EPISODES, tree=env.tree, init=env.init, cost=cost, seed=91,
                                                features=features, verbose=True)
        print(actions_vanilla)